import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..model import HeteroGNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _signed_log1p(value):
    v = float(value)
    if not np.isfinite(v):
        raise ValueError("Non-finite trace value")
    return np.sign(v) * np.log1p(abs(v))


class GraphFolderDataset(Dataset):

    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        self.folder = folder
        self.files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".pt")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt graphs found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        graph = torch.load(path, map_location="cpu", weights_only=False)
        return graph


class GNNPredictor:

    def __init__(self, hidden_channels=64, dropout=0.1, lr=4e-4,
                 loss_weights=(1.0, 0.1, 0.1)):
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.lr = lr
        self.loss_weights = loss_weights

        self.model = None
        self.optimizer = None
        self.device = self._detect_device()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_mae_time': [],
            'val_mae_rem': []
        }

        set_seed()

    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def prepare_data(self, df, test_size=0.3, val_split=0.5):
        from torch_geometric.data import HeteroData
        
        print("\nPreparing data for GNN...")
        split_col = "__split" if "__split" in df.columns else None
        if split_col:
            df = df.copy()
        
        trace_cols = []
        grouped = df.groupby("CaseID")
        for col in df.columns:
            if col in ["CaseID", "Activity", "Resource", "Timestamp", "__split"]:
                continue
            if grouped[col].nunique().max() == 1:
                trace_cols.append(col)
        
        print(f"Detected trace attributes: {trace_cols}")
        
        def build_prefix_df(input_df, scope_name="dataset"):
            rows = []
            total_cases = input_df["CaseID"].nunique()
            processed_cases = 0
            prefix_columns = [
                "CaseID",
                "prefix_id",
                "prefix_pos",
                "prefix_length",
                "Activity",
                "Resource",
                "Timestamp",
                "next_activity",
                "next_timestamp",
                "remaining_time_target",
                *trace_cols,
            ]
            for case_id, group in input_df.groupby("CaseID"):
                processed_cases += 1
                group = group.sort_values("Timestamp").reset_index(drop=True)
                trace_len = len(group)
                trace_attrs = {c: group[c].iloc[0] for c in trace_cols if c in group.columns}

                if trace_len <= 1:
                    continue

                activities = group["Activity"].to_numpy()
                timestamps = group["Timestamp"].to_numpy()
                resources = group["Resource"].to_numpy() if "Resource" in group.columns else None

                for k in range(1, trace_len):
                    label_next_activity = activities[k]
                    next_timestamp = timestamps[k]
                    prefix_current_timestamp = timestamps[k - 1]
                    remaining_time_target = max(
                        0.0,
                        (timestamps[-1] - prefix_current_timestamp).total_seconds(),
                    )
                    for pos in range(k):
                        row = {
                            "CaseID": case_id,
                            "prefix_id": k,
                            "prefix_pos": pos + 1,
                            "prefix_length": k,
                            "Activity": activities[pos],
                            "Resource": resources[pos] if resources is not None else "Unknown",
                            "Timestamp": timestamps[pos],
                            "next_activity": label_next_activity,
                            "next_timestamp": next_timestamp,
                            "remaining_time_target": remaining_time_target,
                        }
                        if trace_attrs:
                            row.update(trace_attrs)
                        rows.append(row)

                if processed_cases % 200 == 0:
                    print(
                        f"[GNN] Prefix build progress: "
                        f"{processed_cases}/{total_cases} cases, "
                        f"{len(rows):,} prefix rows"
                    )

            prefix_df = pd.DataFrame(rows, columns=prefix_columns)
            print(f"Generated {len(prefix_df):,} prefix rows")
            if prefix_df.empty:
                case_lengths = input_df.groupby("CaseID").size()
                single_event_cases = int((case_lengths <= 1).sum()) if not case_lengths.empty else 0
                total_cases = int(case_lengths.shape[0])
                raise ValueError(
                    "No prefixes could be generated for the "
                    f"{scope_name}. GNN next-event prediction requires at least one case "
                    "with two or more events after preprocessing and mapping. "
                    f"Cases in {scope_name}: {total_cases}; "
                    f"cases with <=1 event: {single_event_cases}."
                )
            prefix_df["__ts_log"] = np.log1p(
                prefix_df["Timestamp"].astype("int64") // 1_000_000_000
            ).astype("float32")
            return prefix_df

        base_df = df.drop(columns=[split_col]) if split_col else df
        prefix_df = build_prefix_df(base_df, scope_name="full dataset")
        
        vocabs = {}
        all_activities = set(prefix_df["Activity"].unique().tolist()) | set(prefix_df["next_activity"].unique().tolist())
        values = sorted(all_activities)
        vocabs["Activity"] = {v: i for i, v in enumerate(values)}
        res_vals = sorted(prefix_df["Resource"].unique().tolist())
        vocabs["Resource"] = {v: i for i, v in enumerate(res_vals)}
        
        IGNORE_COLS = {
            "CaseID",
            "prefix_id",
            "prefix_pos",
            "prefix_length",
            "Activity",
            "Resource",
            "Timestamp",
            "next_activity",
            "next_timestamp",
            "remaining_time_target",
            "__ts_log",
        }
        trace_attributes = [col for col in prefix_df.columns if col not in IGNORE_COLS]
        
        for col in trace_attributes:
            if pd.api.types.is_numeric_dtype(prefix_df[col]):
                continue
            vals = sorted(prefix_df[col].fillna("NaN").unique().tolist())
            vocabs[col] = {v: i for i, v in enumerate(vals)}
        self.vocabs = vocabs
        print(f"Vocabularies: Activities={len(vocabs['Activity'])}, Resources={len(vocabs['Resource'])}")
        
        def build_graphs(prefix_df):
            groups = prefix_df.groupby(["CaseID", "prefix_id"])
            graphs = []
            
            print(f"Building {groups.ngroups:,} graphs...")
            for (_, _), p in tqdm(groups, desc="Building graphs", ncols=100):
                p = p.sort_values("prefix_pos")
                
                data = HeteroData()
                k = len(p)
                
                act_map = vocabs["Activity"]
                res_map = vocabs["Resource"]
                
                act_ids = np.array([act_map[a] for a in p["Activity"]])
                data["activity"].x = F.one_hot(torch.tensor(act_ids, dtype=torch.long), num_classes=len(act_map)).float()
                data["activity"].num_nodes = k
                
                res_ids = np.array([res_map[r] for r in p["Resource"]])
                data["resource"].x = F.one_hot(torch.tensor(res_ids, dtype=torch.long), num_classes=len(res_map)).float()
                data["resource"].num_nodes = k
                
                data["time"].x = torch.tensor(p["__ts_log"].to_numpy(), dtype=torch.float32).unsqueeze(1)
                data["time"].num_nodes = k
                
                trace_features = []
                first = p.iloc[0]
                for col in trace_attributes:
                    if col not in p.columns:
                        continue
                    val = first[col]
                    if col in vocabs:
                        idx = vocabs[col].get(val, 0)
                        trace_features.append(F.one_hot(torch.tensor(idx), num_classes=len(vocabs[col])).float())
                    else:
                        try:
                            trace_features.append(torch.tensor([_signed_log1p(val)], dtype=torch.float32))
                        except Exception:
                            trace_features.append(torch.zeros(1))
                
                if not trace_features:
                    trace_features = [torch.zeros(1)]
                data["trace"].x = torch.cat(trace_features).unsqueeze(0)
                data["trace"].num_nodes = 1
                
                idx = torch.arange(k)
                if k > 1:
                    dfr = torch.stack([idx[:-1], idx[1:]])
                else:
                    dfr = torch.empty((2, 0), dtype=torch.long)
                
                data["activity", "next", "activity"].edge_index = dfr
                data["resource", "next", "resource"].edge_index = dfr.clone()
                data["time", "next", "time"].edge_index = dfr.clone()
                
                same_ev = torch.stack([idx, idx])
                data["activity", "same_event", "resource"].edge_index = same_ev
                data["resource", "same_event", "activity"].edge_index = same_ev.clone()
                data["activity", "same_time", "time"].edge_index = same_ev.clone()
                data["time", "same_time", "activity"].edge_index = same_ev.clone()
                
                trace_src = torch.zeros(k, dtype=torch.long)
                data["activity", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
                data["resource", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
                data["time", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
                
                next_act_name = p.iloc[0]["next_activity"]
                if next_act_name not in act_map:
                    print(f"Warning: Activity '{next_act_name}' not in vocabulary. Skipping graph.")
                    continue
                next_act = act_map[next_act_name]
                data.y_activity = torch.tensor([next_act], dtype=torch.long)
                
                t_next = p.iloc[0]["next_timestamp"].timestamp()
                data.y_timestamp = torch.tensor([np.log1p(t_next)], dtype=torch.float32)
                
                remaining = float(p.iloc[0]["remaining_time_target"])
                data.y_remaining_time = torch.tensor([np.log1p(remaining)], dtype=torch.float32)
                
                graphs.append(data)
            
            return graphs
        
        if split_col:
            split_values = set(df[split_col].dropna().unique().tolist())
            if not {"train", "val", "test"}.issubset(split_values):
                raise ValueError("Split column must include train, val, and test values.")

            train_df = df[df[split_col] == "train"].drop(columns=[split_col])
            val_df = df[df[split_col] == "val"].drop(columns=[split_col])
            test_df = df[df[split_col] == "test"].drop(columns=[split_col])

            train_graphs = build_graphs(build_prefix_df(train_df, scope_name="train split"))
            val_graphs = build_graphs(build_prefix_df(val_df, scope_name="validation split"))
            test_graphs = build_graphs(build_prefix_df(test_df, scope_name="test split"))
            graphs = train_graphs + val_graphs + test_graphs
        else:
            graphs = build_graphs(prefix_df)

        print(f"[OK] Built {len(graphs):,} graphs")
        if not graphs:
            raise ValueError(
                "No GNN graphs were built from the generated prefixes. "
                "Check whether the dataset contains valid multi-event cases and a valid activity mapping."
            )
        
        if not split_col:
            n_total = len(graphs)
            n_test = int(n_total * test_size)
            n_train_val = n_total - n_test
            n_val = int(n_train_val * val_split)
            n_train = n_train_val - n_val
            
            train_graphs = graphs[:n_train]
            val_graphs = graphs[n_train:n_train + n_val]
            test_graphs = graphs[n_train + n_val:]
        
        print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
        
        return {
            'train': train_graphs,
            'val': val_graphs,
            'test': test_graphs,
            'sample_graph': graphs[0],
            'num_activity_classes': len(vocabs['Activity']),
            'vocabs': vocabs
        }

    def prepare_data_from_graphs(self, graph_folder, test_size=0.3, val_split=0.5):
        print("\nLoading graph dataset...")
        dataset = GraphFolderDataset(graph_folder)
        print(f"Found {len(dataset)} graphs")

        n = len(dataset)
        test_val_size = int(test_size * n)
        train_size = n - test_val_size
        val_size = int(val_split * test_val_size)
        test_size = test_val_size - val_size

        print(f"\nDataset splits:")
        print(f"Train: {train_size:,} samples")
        print(f"Validation: {val_size:,} samples")
        print(f"Test: {test_size:,} samples")

        train_graphs = [dataset[i] for i in range(train_size)]
        val_graphs = [dataset[i] for i in range(train_size, train_size + val_size)]
        test_graphs = [dataset[i] for i in range(train_size + val_size, n)]

        return {
            'train_graphs': train_graphs,
            'val_graphs': val_graphs,
            'test_graphs': test_graphs,
            'sample_graph': dataset[0]
        }

    def build_model(self, sample_graph, batch_size=64, num_workers=0, num_activity_classes=None):
        print("\nBuilding GNN model...")

        metadata = sample_graph.metadata()
        proj_dims = {k: v.size(-1) for k, v in sample_graph.x_dict.items()}
        
        if num_activity_classes is None:
            num_classes = int(torch.max(sample_graph.y_activity).item()) + 1
        else:
            num_classes = num_activity_classes

        self.model = HeteroGNN(
            metadata=metadata,
            hidden_channels=self.hidden_channels,
            proj_dims=proj_dims,
            num_activity_classes=num_classes,
            dropout=self.dropout,
            loss_weights=self.loss_weights,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        print(f"Model built with {num_classes} activity classes")
        print(f"Using device: {self.device}")

    def create_loaders(self, train_graphs, val_graphs, test_graphs, batch_size=64, num_workers=0):
        loader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list,
        )

        train_loader = torch.utils.data.DataLoader(train_graphs, shuffle=True, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_graphs, shuffle=False, **loader_args)
        test_loader = torch.utils.data.DataLoader(test_graphs, shuffle=False, **loader_args)

        return train_loader, val_loader, test_loader

    def train_epoch(self, loader, log_every=200):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader, 1):
            batch = batch.to(self.device)
            act_logits, time_pred, rem_pred = self.model(batch)
            loss = self.model.compute_loss(act_logits, time_pred, rem_pred, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if log_every and batch_idx % log_every == 0:
                print(f"  [Train] batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

        return total_loss / len(loader)

    def evaluate(self, loader, max_batches=None):
        self.model.eval()
        correct = 0
        total = 0
        mae_time = 0.0
        mae_rem = 0.0
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                act_logits, time_pred, rem_pred = self.model(batch)
                loss = self.model.compute_loss(act_logits, time_pred, rem_pred, batch)
                total_loss += loss.item()
                batches += 1

                y_act = batch.y_activity.view(-1)
                pred = act_logits.argmax(dim=1)
                correct += (pred == y_act).sum().item()
                total += y_act.numel()

                y_time = batch.y_timestamp.view(-1)
                y_rem = batch.y_remaining_time.view(-1)
                mae_time += torch.abs(time_pred - y_time).mean().item()
                mae_rem += torch.abs(rem_pred - y_rem).mean().item()
                if max_batches is not None and batches >= max_batches:
                    break

        return {
            'accuracy': correct / total if total else 0.0,
            'mae_time': mae_time / max(batches, 1),
            'mae_rem': mae_rem / max(batches, 1),
            'loss': total_loss / max(batches, 1),
        }

    def train(self, data, epochs=50, batch_size=64, patience=10, num_workers=0, log_every=200, train_eval_batches=25):
        print(f"\nTraining GNN for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.lr}")

        train_loader, val_loader, _ = self.create_loaders(
            data['train'],
            data['val'],
            data['test'],
            batch_size=batch_size,
            num_workers=num_workers
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, log_every=log_every)
            train_metrics = self.evaluate(train_loader, max_batches=train_eval_batches)
            val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_mae_time'].append(val_metrics['mae_time'])
            self.history['val_mae_rem'].append(val_metrics['mae_rem'])

            print(
                f"Epoch {epoch:03d} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"MAE Time: {val_metrics['mae_time']:.4f} | "
                f"MAE Rem: {val_metrics['mae_rem']:.4f}"
            )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("\nTraining completed!")

    def evaluate_test(self, data, batch_size=64):
        print("\nEvaluating on test set...")

        _, _, test_loader = self.create_loaders(
            data['train'],
            data['val'],
            data['test'],
            batch_size=batch_size
        )

        metrics = self.evaluate(test_loader)

        return metrics

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "gnn_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def plot_training_history(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.history['train_acc'], label='Train')
        axes[0, 0].plot(self.history['val_acc'], label='Validation')
        axes[0, 0].set_title('Activity Prediction Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_loss'], label='Train')
        axes[0, 1].plot(self.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Total Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['val_mae_time'])
        axes[1, 0].set_title('Event Time MAE (Validation)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['val_mae_rem'])
        axes[1, 1].set_title('Remaining Time MAE (Validation)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "gnn_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training history plot saved to: {output_path}")

    def save_results(self, metrics, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "gnn_results.txt")
        with open(results_file, 'w') as f:
            f.write("="*50 + "\n")
            f.write("GNN MODEL - EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Test Accuracy (Activity):  {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Test MAE (Event Time):     {metrics['mae_time']:.4f}\n")
            f.write(f"Test MAE (Remaining Time): {metrics['mae_rem']:.4f}\n")
            f.write(f"Test Loss:                 {metrics['loss']:.4f}\n")
            f.write("\n" + "="*50 + "\n")

        print(f"Results saved to: {results_file}")
