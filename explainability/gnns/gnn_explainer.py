import json
import os
import re
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm
import graphviz

try:
    from torch_geometric.explain import Explainer, GNNExplainer
except ImportError as exc:
    Explainer = None
    GNNExplainer = None
    _PYG_EXPLAIN_IMPORT_ERROR = exc
else:
    _PYG_EXPLAIN_IMPORT_ERROR = None


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})


def _dir_has_png(path):
    if not os.path.isdir(path):
        return False
    return any(name.lower().endswith(".png") for name in os.listdir(path))


def _remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def _serialize_scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _inverse_vocab(vocab):
    if not isinstance(vocab, dict):
        return {}
    return {idx: value for value, idx in vocab.items()}


def _graphviz_safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return cleaned or "node"


class _TaskHeadModel(torch.nn.Module):
    def __init__(self, base_model, task: str):
        super().__init__()
        self.base_model = base_model
        self.task = task

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        act, event_time, remaining_time = self.base_model(
            x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
        )
        if self.task == "activity":
            return act
        if self.task == "event_time":
            return event_time
        if self.task == "remaining_time":
            return remaining_time
        raise ValueError(f"Unsupported task for explanation: {self.task}")


@dataclass
class LocalExplanationArtifacts:
    node_rows: list
    edge_rows: list
    prediction_summary: dict


class ProphetGNNExplainer:
    """
    PROPHET-style explanation wrapper.

    Local explanations come from GNNExplainer node/edge masks.
    Global explanations aggregate node-type scores across explained graphs.
    """

    def __init__(self, model, device, task="activity", vocabularies=None, epochs=200, preferred_time_unit=None):
        if Explainer is None or GNNExplainer is None:
            raise ImportError(
                "torch_geometric.explain is required for PROPHET-style GNN explainability. "
                f"Import failed: {_PYG_EXPLAIN_IMPORT_ERROR}"
            )
        self.model = model
        self.device = device
        self.task = task
        self.vocabularies = vocabularies or {}
        self.epochs = epochs
        self.preferred_time_unit = preferred_time_unit or ("seconds", 1.0, "s")

        self._activity_vocab = _inverse_vocab(self.vocabularies.get("Activity"))
        self._resource_vocab = _inverse_vocab(self.vocabularies.get("Resource"))
        self._explainer = Explainer(
            model=_TaskHeadModel(self.model, task),
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=self._model_config(task),
        )

    @staticmethod
    def _model_config(task):
        if task == "activity":
            return {
                "mode": "multiclass_classification",
                "task_level": "graph",
                "return_type": "raw",
            }
        return {
            "mode": "regression",
            "task_level": "graph",
            "return_type": "raw",
        }

    def _activity_name(self, row):
        idx = int(np.argmax(row))
        return self._activity_vocab.get(idx, f"activity_{idx}")

    def _resource_name(self, row):
        idx = int(np.argmax(row))
        return self._resource_vocab.get(idx, f"resource_{idx}")

    def _has_informative_resource_view(self):
        labels = {
            str(value).strip().lower()
            for value in self._resource_vocab.values()
            if str(value).strip()
        }
        placeholder_labels = {"unknown", "nan", "none", "null", "n/a", "na"}
        return bool(labels) and not labels.issubset(placeholder_labels)

    @staticmethod
    def _decode_log_seconds(value):
        return float(np.expm1(float(value)))

    @staticmethod
    def choose_preferred_time_unit(seconds_values):
        values = [float(v) for v in seconds_values if np.isfinite(v) and float(v) >= 0]
        if not values:
            return ("seconds", 1.0, "s")
        return ("compound", 1.0, "")

    @staticmethod
    def _format_compound_duration(seconds_value):
        seconds = max(0.0, float(seconds_value))
        total_seconds = int(round(seconds))
        if total_seconds == 0:
            return "0 s"

        minute = 60
        hour = 60 * minute
        day = 24 * hour
        month = 30 * day

        if total_seconds >= month:
            months, remainder = divmod(total_seconds, month)
            days = remainder // day
            return f"{months} mo {days} d" if days else f"{months} mo"
        if total_seconds >= day:
            days, remainder = divmod(total_seconds, day)
            hours = remainder // hour
            return f"{days} d {hours} h" if hours else f"{days} d"
        if total_seconds >= hour:
            hours, remainder = divmod(total_seconds, hour)
            minutes = remainder // minute
            return f"{hours} h {minutes} min" if minutes else f"{hours} h"
        if total_seconds >= minute:
            minutes, remainder = divmod(total_seconds, minute)
            return f"{minutes} min {remainder} s" if remainder else f"{minutes} min"
        return f"{total_seconds} s"

    def _format_duration(self, seconds_value, decimals=2):
        unit_name, divisor, suffix = getattr(self, "preferred_time_unit", ("compound", 1.0, ""))
        if unit_name == "compound":
            return self._format_compound_duration(seconds_value)
        scaled = float(seconds_value) / float(divisor)
        return f"{scaled:.{decimals}f} {suffix}"

    def _regression_question(self):
        if self.task == "event_time":
            return "Which graph components most influenced the predicted time until the next event?"
        if self.task == "remaining_time":
            return "Which graph components most influenced the predicted time until process completion?"
        return "Which graph components most influenced the predicted time outcome?"

    @staticmethod
    def _graph_metadata_value(graph, name):
        value = getattr(graph, name, None)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return str(value.detach().cpu().view(-1)[0].item())
            return None
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return str(value[0])
        return str(value)

    @staticmethod
    def _case_title_line(prediction):
        case_id = prediction.get("case_id")
        if case_id is None or str(case_id).strip() == "":
            return None
        return f"Case ID: {case_id}"

    def _node_label(self, graph, node_type, node_idx):
        x = graph[node_type].x[node_idx].detach().cpu().numpy()
        if node_type == "activity":
            return self._activity_name(x)
        if node_type == "resource":
            return self._resource_name(x)
        if node_type == "time":
            display_seconds = getattr(graph["time"], "display_seconds", None)
            if display_seconds is not None and int(node_idx) < len(display_seconds):
                seconds_value = float(display_seconds[node_idx].detach().cpu().item())
                return f"t{node_idx + 1}: +{self._format_duration(seconds_value, decimals=1)}"
            return f"time_{node_idx + 1}"
        if node_type == "trace":
            return "Case context"
        return f"{node_type}_{node_idx}"

    def _prediction_summary(self, graph):
        self.model.eval()
        with torch.no_grad():
            act, event_time, remaining_time = self.model(graph)
            graph_metadata = {
                "case_id": self._graph_metadata_value(graph, "case_id"),
                "prefix_id": self._graph_metadata_value(graph, "prefix_id"),
            }
            if self.task == "activity":
                probs = torch.softmax(act, dim=-1)
                pred_idx = int(probs.argmax(dim=-1).item())
                true_idx = int(graph.y_activity.view(-1)[0].item())
                return {
                    **graph_metadata,
                    "task": self.task,
                    "predicted_index": pred_idx,
                    "predicted_label": self._activity_vocab.get(pred_idx, str(pred_idx)),
                    "true_index": true_idx,
                    "true_label": self._activity_vocab.get(true_idx, str(true_idx)),
                    "confidence": float(probs[0, pred_idx].item()),
                }
            if self.task == "event_time":
                prediction_log_seconds = float(event_time.view(-1)[0].item())
                target_log_seconds = float(graph.y_timestamp.view(-1)[0].item())
                prediction_seconds = self._decode_log_seconds(prediction_log_seconds)
                target_seconds = self._decode_log_seconds(target_log_seconds)
                return {
                    **graph_metadata,
                    "task": self.task,
                    "prediction_log_seconds": prediction_log_seconds,
                    "target_log_seconds": target_log_seconds,
                    "prediction_seconds": prediction_seconds,
                    "target_seconds": target_seconds,
                    "prediction_display": self._format_duration(prediction_seconds),
                    "target_display": self._format_duration(target_seconds),
                    "display_unit": self.preferred_time_unit[0],
                    "display_suffix": self.preferred_time_unit[2],
                    "explanation_question": self._regression_question(),
                }
            prediction_log_seconds = float(remaining_time.view(-1)[0].item())
            target_log_seconds = float(graph.y_remaining_time.view(-1)[0].item())
            prediction_seconds = self._decode_log_seconds(prediction_log_seconds)
            target_seconds = self._decode_log_seconds(target_log_seconds)
            return {
                **graph_metadata,
                "task": self.task,
                "prediction_log_seconds": prediction_log_seconds,
                "target_log_seconds": target_log_seconds,
                "prediction_seconds": prediction_seconds,
                "target_seconds": target_seconds,
                "prediction_display": self._format_duration(prediction_seconds),
                "target_display": self._format_duration(target_seconds),
                "display_unit": self.preferred_time_unit[0],
                "display_suffix": self.preferred_time_unit[2],
                "explanation_question": self._regression_question(),
            }

    def _format_prediction_summary_lines(self, prediction):
        if prediction["task"] == "activity":
            return [
                f"Task: next activity classification",
                f"Predicted: {prediction['predicted_label']}",
                f"True: {prediction['true_label']}",
                f"Confidence: {prediction['confidence']:.3f}",
            ]
        question = prediction.get("explanation_question", self._regression_question())
        return [
            f"Task: {prediction['task'].replace('_', ' ')} regression",
            question,
            f"Predicted time: {prediction['prediction_display']}",
            f"Actual time: {prediction['target_display']}",
        ]

    def _sample_description(self, artifacts, max_nodes=5, max_edges=5):
        prediction = artifacts.prediction_summary
        lines = self._format_prediction_summary_lines(prediction)

        node_df = pd.DataFrame(artifacts.node_rows)
        edge_df = pd.DataFrame(artifacts.edge_rows)

        if not node_df.empty:
            top_nodes = node_df.sort_values("score", ascending=False).head(max_nodes)
            node_parts = [
                f"{row.label} [{row.node_type}] ({row.score:.3f})"
                for row in top_nodes.itertuples()
            ]
            lines.append("Most influential nodes: " + "; ".join(node_parts))

        if not edge_df.empty:
            top_edges = edge_df.sort_values("score", ascending=False).head(max_edges)
            edge_parts = [
                f"{row.source_label} -> {row.target_label} [{row.edge_type}] ({row.score:.3f})"
                for row in top_edges.itertuples()
            ]
            lines.append("Most influential relations: " + "; ".join(edge_parts))

        if prediction["task"] == "activity":
            lines.append(
                "Interpretation: the model predicts the next activity based primarily on the influential nodes and connections shown. For a human user, this means the highlighted sequence of events (nodes) and their relationships (edges) are the strongest indicators of what the system expects to happen next."
            )
        else:
            if prediction["task"] == "event_time":
                lines.append(
                    "Interpretation: the model predicts the time until the next event based on these highlighted graph components. For a human user, this shows which past events and connections are most responsible for accelerating or delaying the upcoming step."
                )
            else:
                lines.append(
                    "Interpretation: the model predicts the time until process completion based on these highlighted graph components. For a human user, this shows which sequence of past events and relationships are driving the overall timeline of the process."
                )

        return lines

    def _graphviz_activity_caption(self, edge_row):
        relation = str(edge_row.relation)
        if relation == "next":
            return "follow"
        if {edge_row.source_type, edge_row.target_type} == {"activity", "resource"}:
            return "perform"
        if {edge_row.source_type, edge_row.target_type} == {"activity", "time"}:
            return "timing"
        if "trace" in {edge_row.source_type, edge_row.target_type}:
            return "context"
        return "link"

    def _short_activity_label(self, label, max_len=18):
        text = str(label)
        if len(text) <= max_len:
            return text
        words = [word for word in text.replace("_", " ").split() if word]
        if len(words) >= 2:
            initials = "".join(word[0].upper() for word in words[:4])
            if len(initials) >= 2:
                return initials
        return text[: max_len - 1] + "…"

    def _short_graphviz_label(self, label, node_type):
        text = str(label).strip()
        if node_type == "activity":
            words = [word for word in text.replace("_", " ").split() if word.lower() not in {"by", "the", "of", "and"}]
            if not words:
                return "ACT"
            initials = "".join(word[0].upper() for word in words[:4])
            return initials if len(initials) >= 2 else self._short_activity_label(text, max_len=6)
        if node_type == "resource":
            upper = [word[0].upper() for word in text.replace("_", " ").split() if word]
            return "".join(upper[:3]) or "RES"
        if node_type == "time":
            if ":" in text:
                return text.split(":", 1)[0].upper()
            return "T"
        if node_type == "trace":
            return "CTX"
        return text[:4].upper() or "N"

    def _graphviz_box_label(self, text, border_color, bold=False, font_color="#111827"):
        safe = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<BR/>")
        )
        tag_open = "<B>" if bold else ""
        tag_close = "</B>" if bold else ""
        return f'''<
        <TABLE BORDER="1" COLOR="{border_color}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2" BGCOLOR="white">
            <TR><TD><FONT POINT-SIZE="10" COLOR="{font_color}">{tag_open}{safe}{tag_close}</FONT></TD></TR>
        </TABLE>
        >'''

    def _graphviz_escape(self, value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<BR/>")
        )

    def _graphviz_event_label(self, event_idx, activity_row, resource_row=None, time_row=None):
        activity = self._graphviz_escape(activity_row.label)
        activity_score = float(activity_row.score)
        resource = self._graphviz_escape(resource_row.label) if resource_row is not None else "N/A"
        resource_score = float(resource_row.score) if resource_row is not None else 0.0
        timestamp = self._graphviz_escape(self._time_label_for_plot(time_row.label)) if time_row is not None else "N/A"
        time_score = float(time_row.score) if time_row is not None else 0.0
        return f'''<
        <TABLE BORDER="1" COLOR="#cbd5e1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="white">
            <TR>
                <TD BGCOLOR="#f8fafc" COLSPAN="2">
                    <FONT POINT-SIZE="14" COLOR="#475569"><B>E{event_idx + 1}</B></FONT>
                </TD>
            </TR>
            <TR>
                <TD BGCOLOR="#fff1f2"><FONT POINT-SIZE="12" COLOR="#991b1b"><B>Activity</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{activity}</B><BR/>{activity_score:.2f}</FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#eff6ff"><FONT POINT-SIZE="12" COLOR="#1d4ed8"><B>Resource</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827">{resource}<BR/>{resource_score:.2f}</FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#ecfdf5"><FONT POINT-SIZE="12" COLOR="#047857"><B>Time</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827">{timestamp}<BR/>{time_score:.2f}</FONT></TD>
            </TR>
        </TABLE>
        >'''

    def _time_label_for_plot(self, label):
        text = str(label)
        match = re.match(r"^(t\d+): \+([0-9]+(?:\.[0-9]+)?) s$", text)
        if not match:
            return text
        step, seconds_text = match.groups()
        return f"{step}: +{self._format_duration(float(seconds_text), decimals=1)}"

    def _graphviz_output_label(self, prediction):
        if prediction.get("task") in {"event_time", "remaining_time"}:
            task_label = (
                "Event Time Prediction"
                if prediction.get("task") == "event_time"
                else "Remaining Time Prediction"
            )
            predicted_label = (
                "Predicted event time"
                if prediction.get("task") == "event_time"
                else "Predicted remaining time"
            )
            actual_label = (
                "Actual event time"
                if prediction.get("task") == "event_time"
                else "Actual remaining time"
            )
            predicted = self._graphviz_escape(prediction["prediction_display"])
            actual = self._graphviz_escape(prediction["target_display"])
            error_seconds = abs(float(prediction["prediction_seconds"]) - float(prediction["target_seconds"]))
            error_display = self._graphviz_escape(self._format_duration(error_seconds))
            return f'''<
        <TABLE BORDER="1" COLOR="#64748b" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="white">
            <TR>
                <TD BGCOLOR="#f8fafc" COLSPAN="2">
                    <FONT POINT-SIZE="14" COLOR="#475569"><B>Model Output</B></FONT>
                </TD>
            </TR>
            <TR>
                <TD BGCOLOR="#f8fafc"><FONT POINT-SIZE="12" COLOR="#475569"><B>Task</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{task_label}</B></FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#dbeafe"><FONT POINT-SIZE="12" COLOR="#1d4ed8"><B>{predicted_label}</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{predicted}</B></FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#f8fafc"><FONT POINT-SIZE="12" COLOR="#475569"><B>{actual_label}</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{actual}</B></FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#fff7ed"><FONT POINT-SIZE="12" COLOR="#c2410c"><B>Absolute error</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827">{error_display}</FONT></TD>
            </TR>
        </TABLE>
        >'''

        predicted = self._graphviz_escape(prediction["predicted_label"])
        true = self._graphviz_escape(prediction["true_label"])
        confidence = float(prediction["confidence"])
        return f'''<
        <TABLE BORDER="1" COLOR="#64748b" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="white">
            <TR>
                <TD BGCOLOR="#f8fafc" COLSPAN="2">
                    <FONT POINT-SIZE="14" COLOR="#475569"><B>Model Output</B></FONT>
                </TD>
            </TR>
            <TR>
                <TD BGCOLOR="#dbeafe"><FONT POINT-SIZE="12" COLOR="#1d4ed8"><B>Predicted next</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{predicted}</B></FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#f8fafc"><FONT POINT-SIZE="12" COLOR="#475569"><B>Ground truth</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827"><B>{true}</B></FONT></TD>
            </TR>
            <TR>
                <TD BGCOLOR="#eff6ff"><FONT POINT-SIZE="12" COLOR="#1d4ed8"><B>Confidence</B></FONT></TD>
                <TD><FONT POINT-SIZE="12" COLOR="#111827">{confidence:.3f}</FONT></TD>
            </TR>
        </TABLE>
        >'''

    def _append_explanation_weight_footer(self, output_path, dpi=180):
        note = "Numerical values represent post-hoc explanation weights for nodes and edges in the explanation graph."
        image = plt.imread(output_path)
        height, width = image.shape[:2]
        footer_height = 220
        total_height = height + footer_height

        fig = plt.figure(figsize=(width / dpi, total_height / dpi), dpi=dpi, facecolor="white")
        image_ax = fig.add_axes([0, footer_height / total_height, 1, height / total_height])
        image_ax.imshow(image)
        image_ax.set_axis_off()

        footer_ax = fig.add_axes([0, 0, 1, footer_height / total_height])
        footer_ax.set_axis_off()
        footer_ax.text(
            0.5,
            0.5,
            note,
            ha="center",
            va="center",
            fontsize=26,
            color="#334155",
            bbox={
                "boxstyle": "round,pad=0.65",
                "fc": "#f8fafc",
                "ec": "#cbd5e1",
                "alpha": 1.0,
            },
        )

        fig.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor="white")
        plt.close(fig)

    def _draw_local_dfg_graphviz(self, artifacts, output_path):
        prediction = artifacts.prediction_summary
        task = prediction.get("task")
        if task not in {"activity", "event_time", "remaining_time"}:
            return False

        node_df = pd.DataFrame(artifacts.node_rows)
        edge_df = pd.DataFrame(artifacts.edge_rows)
        activity_df = node_df[node_df["node_type"] == "activity"].sort_values("node_index").reset_index(drop=True)
        resource_df = node_df[node_df["node_type"] == "resource"].sort_values("node_index").reset_index(drop=True)
        time_df = node_df[node_df["node_type"] == "time"].sort_values("node_index").reset_index(drop=True)
        if activity_df.empty:
            return False

        dot = graphviz.Digraph(comment="Local DFG Trace Sequence")
        dot.attr(
            rankdir="LR",
            splines="spline",
            nodesep="0.45",
            ranksep="0.7",
            pad="0.2",
            margin="0.05",
            concentrate="false",
            ordering="out",
            dpi="180",
        )
        title_lines = ["Local Explanation - DFG Trace Sequence"]
        case_title = self._case_title_line(prediction)
        if case_title:
            title_lines.append(case_title)
        if task == "activity":
            title_lines.append(
                f"Predicted: {prediction['predicted_label']} | "
                f"True: {prediction['true_label']} | "
                f"Confidence: {prediction['confidence']:.3f}"
            )
        elif task == "event_time":
            title_lines.append(
                "Task: event time regression | "
                f"Predicted event time: {prediction['prediction_display']} | "
                f"Actual event time: {prediction['target_display']}"
            )
        else:
            title_lines.append(
                "Task: remaining time regression | "
                f"Predicted remaining time: {prediction['prediction_display']} | "
                f"Actual remaining time: {prediction['target_display']}"
            )
        dot.attr(
            label="\n".join(title_lines),
            labelloc="t",
            fontsize="24",
            fontname="Helvetica",
            fontcolor="#0f172a",
        )
        dot.attr("node", shape="plain", fontname="Helvetica")
        dot.attr("edge", fontname="Helvetica", fontsize="12", color="#0f172a", arrowsize="0.9")

        event_count = int(activity_df["node_index"].max()) + 1
        event_ids = []
        for idx in range(event_count):
            activity_rows = activity_df[activity_df["node_index"] == idx]
            if activity_rows.empty:
                continue
            resource_rows = resource_df[resource_df["node_index"] == idx]
            time_rows = time_df[time_df["node_index"] == idx]
            node_id = f"event_{idx}"
            event_ids.append(node_id)
            dot.node(
                node_id,
                label=self._graphviz_event_label(
                    idx,
                    activity_rows.iloc[0],
                    resource_rows.iloc[0] if not resource_rows.empty else None,
                    time_rows.iloc[0] if not time_rows.empty else None,
                ),
            )

        next_scores = {}
        if not edge_df.empty:
            next_edges = edge_df[edge_df["relation"] == "next"]
            for row in next_edges.itertuples():
                if row.source_type == "activity" and row.target_type == "activity":
                    next_scores[(int(row.source_index), int(row.target_index))] = float(row.score)

        for idx in range(event_count - 1):
            src_id = f"event_{idx}"
            dst_id = f"event_{idx + 1}"
            if src_id not in event_ids or dst_id not in event_ids:
                continue
            score = next_scores.get((idx, idx + 1))
            edge_label = "" if score is None else f"next {score:.2f}"
            dot.edge(
                src_id,
                dst_id,
                label=edge_label,
                color="#0891b2",
                penwidth="2.0",
                weight="8",
                constraint="true",
            )

        output_node = "model_output"
        dot.node(output_node, label=self._graphviz_output_label(prediction))

        last_event_id = f"event_{event_count - 1}"
        if last_event_id in event_ids:
            output_edge_label = ""
            if task == "event_time":
                output_edge_label = "predict event time"
            elif task == "remaining_time":
                output_edge_label = "predict remaining time"
            dot.edge(
                last_event_id,
                output_node,
                label=output_edge_label,
                color="#1d4ed8",
                style="dashed",
                penwidth="2.0",
                constraint="true",
            )

        base_path, _ = os.path.splitext(output_path)
        dot.render(base_path, format="png", cleanup=True)
        self._append_explanation_weight_footer(output_path)
        return True

    def _draw_local_graphviz_activity(self, artifacts, output_path):
        prediction = artifacts.prediction_summary
        if prediction.get("task") != "activity":
            return

        node_df = pd.DataFrame(artifacts.node_rows)
        edge_df = pd.DataFrame(artifacts.edge_rows)
        activity_df = node_df[node_df["node_type"] == "activity"].sort_values("node_index").reset_index(drop=True)
        resource_df = node_df[node_df["node_type"] == "resource"].sort_values("node_index").reset_index(drop=True)
        time_df = node_df[node_df["node_type"] == "time"].sort_values("node_index").reset_index(drop=True)
        trace_df = node_df[node_df["node_type"] == "trace"].sort_values("node_index").reset_index(drop=True)
        if activity_df.empty:
            return

        dot = graphviz.Digraph(comment="Local Activity Flow")
        dot.attr(
            rankdir="LR",
            splines="ortho",
            nodesep="0.45",
            ranksep="0.9",
            pad="0.15",
            margin="0.05",
            ratio="compress",
            concentrate="false",
        )
        title_lines = ["Local Explanation"]
        case_title = self._case_title_line(prediction)
        if case_title:
            title_lines.append(case_title)
        title_lines.append(
            f"Predicted: {prediction['predicted_label']} | "
            f"True: {prediction['true_label']} | "
            f"Confidence: {prediction['confidence']:.3f}"
        )
        dot.attr(
            label="\n".join(title_lines),
            labelloc="t",
            fontsize="18",
            fontname="Helvetica",
        )
        dot.attr("node", shape="circle", style="filled", fixedsize="true", width="0.26", height="0.26", label="")
        dot.attr("edge", fontname="Helvetica", fontsize="10")

        node_ids = {}
        all_scores = node_df["score"].tolist() if not node_df.empty else [0.0]
        max_score = max(float(max(all_scores)), 1e-6)

        def add_node(node_type, row):
            node_id = f"{node_type}_{int(row.node_index)}"
            node_ids[(node_type, int(row.node_index))] = node_id
            intensity = float(row.score) / max_score
            if node_type == "activity":
                fill = "#ef5350" if intensity >= 0.7 else "#f28b82" if intensity >= 0.35 else "#f8b4b4"
            elif node_type == "resource":
                fill = "#7986cb" if intensity >= 0.7 else "#9fa8da" if intensity >= 0.35 else "#c5cae9"
            elif node_type == "time":
                fill = "#4dd0e1" if intensity >= 0.7 else "#80deea" if intensity >= 0.35 else "#b2ebf2"
            else:
                fill = "#b39ddb" if intensity >= 0.7 else "#d1c4e9" if intensity >= 0.35 else "#ede7f6"
            short = self._short_graphviz_label(row.label, node_type)
            score_text = f"{short}: {row.score:.2f}"
            dot.node(node_id, color=fill, fillcolor=fill, xlabel=score_text)

        if not trace_df.empty:
            trace_row = trace_df.iloc[0]
            add_node("trace", trace_row)

        event_count = int(max(activity_df["node_index"].max(), resource_df["node_index"].max() if not resource_df.empty else 0, time_df["node_index"].max() if not time_df.empty else 0)) + 1
        for idx in range(event_count):
            with dot.subgraph(name=f"cluster_evt_{idx}") as sg:
                sg.attr(rank="same")
                if idx < len(activity_df):
                    add_node("activity", activity_df.iloc[idx])
                    sg.node(node_ids[("activity", idx)])
                if idx < len(resource_df):
                    add_node("resource", resource_df.iloc[idx])
                    sg.node(node_ids[("resource", idx)])
                if idx < len(time_df):
                    add_node("time", time_df.iloc[idx])
                    sg.node(node_ids[("time", idx)])

        for idx in range(event_count - 1):
            for node_type in ("activity", "resource", "time"):
                left_id = node_ids.get((node_type, idx))
                right_id = node_ids.get((node_type, idx + 1))
                if left_id and right_id:
                    dot.edge(left_id, right_id, color="transparent", style="invis", weight="40")

        relation_color = {
            "follow": "#4dd0e1",
            "perform": "#ef5350",
            "timing": "#4dd0e1",
            "context": "#9aa9bf",
            "link": "#9aa9bf",
        }

        next_edges = edge_df[edge_df["relation"] == "next"].sort_values("score", ascending=False)
        for row in next_edges.itertuples():
            src_id = node_ids.get((row.source_type, int(row.source_index)))
            dst_id = node_ids.get((row.target_type, int(row.target_index)))
            if not src_id or not dst_id:
                continue
            caption = self._graphviz_activity_caption(row)
            score = float(row.score)
            dot.edge(
                src_id,
                dst_id,
                label=self._graphviz_box_label(f"{caption}:{score:.2f}", relation_color[caption], bold=score >= 0.75),
                color=relation_color[caption],
                penwidth="1.5",
                constraint="true",
            )

        candidate_edges = edge_df.sort_values("score", ascending=False)
        added_keys = set()
        added_count = 0
        for row in candidate_edges.itertuples():
            src_key = (row.source_type, int(row.source_index))
            dst_key = (row.target_type, int(row.target_index))
            src_id = node_ids.get(src_key)
            dst_id = node_ids.get(dst_key)
            if not src_id or not dst_id:
                continue
            key = (src_key, dst_key, row.relation)
            if key in added_keys or row.relation == "next":
                continue
            added_keys.add(key)
            caption = self._graphviz_activity_caption(row)
            score = float(row.score)
            dot.edge(
                src_id,
                dst_id,
                label=self._graphviz_box_label(
                    f"{caption}:{score:.2f}",
                    relation_color.get(caption, "#9aa9bf"),
                    bold=score >= 0.75,
                    font_color="red" if caption == "perform" else "#111827",
                ),
                color=relation_color.get(caption, "#9aa9bf"),
                penwidth="1.35",
                constraint="false",
            )
            added_count += 1
            if added_count >= 22:
                break

        dot.attr("node", shape="box", style="rounded,filled", fixedsize="false", width="0", height="0", label="")
        pred_node = "predicted_output"
        gt_node = "ground_truth_output"
        dot.node(
            pred_node,
            label=f"Predicted next: {prediction['predicted_label']}",
            fillcolor="#dbeafe",
            color="#1d4ed8",
            fontcolor="#0f172a",
        )
        dot.node(
            gt_node,
            label=f"Ground truth: {prediction['true_label']}",
            fillcolor="#f8fafc",
            color="#64748b",
            fontcolor="#0f172a",
        )
        last_activity_id = node_ids.get(("activity", event_count - 1))
        if last_activity_id:
            dot.edge(last_activity_id, pred_node, color="#1d4ed8", style="dashed", penwidth="1.5")
            dot.edge(last_activity_id, gt_node, color="#64748b", style="dotted", penwidth="1.5")

        base_path, _ = os.path.splitext(output_path)
        dot.render(base_path, format="png", cleanup=True)

    def _non_empty_edge_index_dict(self, graph):
        filtered = {}
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_index is None:
                continue
            if edge_index.numel() == 0:
                continue
            if edge_index.dim() != 2 or edge_index.size(1) == 0:
                continue
            filtered[edge_type] = edge_index
        return filtered

    def explain_graph(self, graph):
        graph = graph.to(self.device)
        prediction = self._prediction_summary(graph)

        target = None
        if self.task == "activity":
            target = prediction["predicted_index"]

        edge_index_dict = self._non_empty_edge_index_dict(graph)
        if not edge_index_dict:
            raise RuntimeError(
                "PROPHET explainability failed: the graph has no non-empty edge relations to explain."
            )

        explanation = self._explainer(
            graph.x_dict,
            edge_index_dict,
            target=target,
        )
        return explanation, prediction

    def summarize_local_explanation(self, graph, explanation, prediction):
        node_rows = []
        for node_type in graph.node_types:
            node_mask = getattr(explanation[node_type], "node_mask", None)
            if node_mask is None:
                continue
            node_mask = node_mask.detach().cpu().numpy()
            if node_mask.ndim > 1:
                node_scores = np.abs(node_mask).mean(axis=1)
            else:
                node_scores = np.abs(node_mask)
            for idx, score in enumerate(node_scores):
                node_rows.append(
                    {
                        "node_type": node_type,
                        "node_index": idx,
                        "label": self._node_label(graph, node_type, idx),
                        "score": float(score),
                    }
                )

        edge_rows = []
        explanation_edge_types = getattr(explanation, "edge_types", [])
        for edge_type in explanation_edge_types:
            edge_mask = getattr(explanation[edge_type], "edge_mask", None)
            if edge_mask is None:
                continue
            edge_mask = edge_mask.detach().cpu().numpy().reshape(-1)
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            rel_name = edge_type[1]
            for idx, score in enumerate(edge_mask):
                src_idx = int(edge_index[0, idx])
                dst_idx = int(edge_index[1, idx])
                edge_rows.append(
                    {
                        "edge_type": f"{edge_type[0]}->{rel_name}->{edge_type[2]}",
                        "relation": rel_name,
                        "source_type": edge_type[0],
                        "target_type": edge_type[2],
                        "source_index": src_idx,
                        "target_index": dst_idx,
                        "source_label": self._node_label(graph, edge_type[0], src_idx),
                        "target_label": self._node_label(graph, edge_type[2], dst_idx),
                        "score": float(abs(score)),
                    }
                )

        return LocalExplanationArtifacts(
            node_rows=node_rows,
            edge_rows=edge_rows,
            prediction_summary=prediction,
        )

    def _draw_local_hetero_graph(self, artifacts, output_path):
        if self._draw_local_dfg_graphviz(artifacts, output_path):
            return

        node_df = pd.DataFrame(artifacts.node_rows)
        edge_df = pd.DataFrame(artifacts.edge_rows)
        prediction = artifacts.prediction_summary

        node_colors = {
            "activity": "#ef4444",
            "resource": "#3b82f6",
            "time": "#10b981",
            "trace": "#8b5cf6",
        }
        relation_colors = {
            "next": "#06b6d4",
            "same_event": "#c75a22",
            "same_time": "#c75a22",
            "to_trace": "#94a3b8",
            "belongs_to": "#94a3b8",
            "has": "#94a3b8",
        }
        row_y = {"trace": 3.55, "activity": 2.6, "resource": 1.55, "time": 0.45}
        node_df = node_df.sort_values(["node_type", "node_index"]).reset_index(drop=True)
        event_count = max(
            int(node_df[node_df["node_type"] == "activity"]["node_index"].max() + 1) if (node_df["node_type"] == "activity").any() else 0,
            int(node_df[node_df["node_type"] == "resource"]["node_index"].max() + 1) if (node_df["node_type"] == "resource").any() else 0,
            int(node_df[node_df["node_type"] == "time"]["node_index"].max() + 1) if (node_df["node_type"] == "time").any() else 0,
        )
        event_count = max(event_count, 1)
        max_node_label_len = max((len(str(row.label)) for row in node_df.itertuples()), default=8)
        event_gap = max(1.75, min(2.35, 0.9 + 0.055 * max_node_label_len))
        graph_left = 1.65
        x_positions = graph_left + np.arange(event_count, dtype=float) * event_gap
        trace_x = float(np.mean(x_positions))
        output_x = float(x_positions[-1]) + 2.0
        x_max = output_x + 1.55
        fig_width = max(18.0, min(34.0, 4.5 + 0.8 * x_max))

        fig = plt.figure(figsize=(fig_width, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.35, 1.25], hspace=0.14)
        ax_graph = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])

        pos = {}
        graph_nx = nx.DiGraph()
        for row in node_df.itertuples():
            node_id = f"{row.node_type}:{row.node_index}"
            if row.node_type == "trace":
                pos[node_id] = (trace_x, row_y["trace"])
            else:
                x = float(x_positions[min(int(row.node_index), event_count - 1)])
                pos[node_id] = (x, row_y.get(row.node_type, 0.5))
            graph_nx.add_node(
                node_id,
                label=row.label,
                score=float(row.score),
                node_type=row.node_type,
            )

        edge_df = edge_df.sort_values("score", ascending=False).reset_index(drop=True)
        edge_limit = min(14, len(edge_df))
        display_edges = edge_df.head(edge_limit).copy()
        if not display_edges.empty:
            non_trace_edges = display_edges[
                ~((display_edges["source_type"] == "trace") | (display_edges["target_type"] == "trace"))
            ]
            if len(non_trace_edges) < min(8, len(display_edges)):
                extra_non_trace = edge_df[
                    ~((edge_df["source_type"] == "trace") | (edge_df["target_type"] == "trace"))
                ].head(6)
                display_edges = (
                    pd.concat([display_edges, extra_non_trace], ignore_index=True)
                    .drop_duplicates(
                        subset=["source_type", "source_index", "target_type", "target_index", "relation"]
                    )
                    .sort_values("score", ascending=False)
                    .head(max(edge_limit, 6))
                )

        for row in display_edges.itertuples():
            src_id = f"{row.source_type}:{row.source_index}"
            dst_id = f"{row.target_type}:{row.target_index}"
            if src_id not in pos or dst_id not in pos:
                continue
            graph_nx.add_edge(
                src_id,
                dst_id,
                relation=row.relation,
                score=float(row.score),
            )

        for step_idx, x in enumerate(x_positions, start=1):
            ax_graph.text(x, 4.1, f"E{step_idx}", ha="center", va="center", fontsize=10, weight="bold", color="#475569")
            ax_graph.axvline(x=x, ymin=0.09, ymax=0.9, color="#e2e8f0", linewidth=0.8, zorder=0)

        row_label_x = 1.05
        ax_graph.text(row_label_x, row_y["trace"], "Case context", va="center", ha="right", fontsize=10, color="#475569", weight="bold")
        ax_graph.text(row_label_x, row_y["activity"], "Activity", va="center", ha="right", fontsize=10, color="#475569", weight="bold")
        ax_graph.text(row_label_x, row_y["resource"], "Resource", va="center", ha="right", fontsize=10, color="#475569", weight="bold")
        ax_graph.text(row_label_x, row_y["time"], "Time", va="center", ha="right", fontsize=10, color="#475569", weight="bold")

        def wrapped_node_label(attrs):
            label_width = 16 if attrs["node_type"] == "activity" else 18
            if attrs["node_type"] == "time":
                label_width = 20
            return textwrap.fill(
                str(attrs["label"]),
                width=label_width,
                break_long_words=False,
                break_on_hyphens=False,
            )

        def node_box_radii(node_id):
            attrs = graph_nx.nodes[node_id]
            wrapped = wrapped_node_label(attrs)
            label_lines = wrapped.splitlines() + [f"{float(attrs['score']):.2f}"]
            max_line_len = max((len(line) for line in label_lines), default=6)
            rx = max(0.34, min(0.78, 0.055 * max_line_len + 0.08))
            ry = 0.14 + 0.08 * max(1, len(label_lines))
            return rx, ry

        def edge_endpoints(src_id, dst_id):
            src_x, src_y = pos[src_id]
            dst_x, dst_y = pos[dst_id]
            dx = dst_x - src_x
            dy = dst_y - src_y
            distance = float(np.hypot(dx, dy))
            if distance <= 1e-6:
                return (src_x, src_y), (dst_x, dst_y)
            ux = dx / distance
            uy = dy / distance

            def boundary_distance(node_id):
                rx, ry = node_box_radii(node_id)
                tx = rx / abs(ux) if abs(ux) > 1e-6 else np.inf
                ty = ry / abs(uy) if abs(uy) > 1e-6 else np.inf
                return min(tx, ty)

            src_shrink = min(boundary_distance(src_id), distance * 0.42)
            dst_shrink = min(boundary_distance(dst_id), distance * 0.42)
            return (
                (src_x + ux * src_shrink, src_y + uy * src_shrink),
                (dst_x - ux * dst_shrink, dst_y - uy * dst_shrink),
            )

        if graph_nx.number_of_edges() > 0:
            edge_scores = [graph_nx.edges[e]["score"] for e in graph_nx.edges()]
            max_edge = max(edge_scores) if edge_scores else 1.0
            for edge in graph_nx.edges():
                relation = graph_nx.edges[edge]["relation"]
                src_x, src_y = pos[edge[0]]
                dst_x, dst_y = pos[edge[1]]
                if relation == "next" and abs(src_y - dst_y) < 0.02:
                    rad = 0.0
                elif "trace" in relation or graph_nx.nodes[edge[0]]["node_type"] == "trace" or graph_nx.nodes[edge[1]]["node_type"] == "trace":
                    rad = -0.08
                else:
                    rad = 0.0
                start, end = edge_endpoints(edge[0], edge[1])
                ax_graph.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-|>",
                        mutation_scale=14,
                        linewidth=1.4 + 5.2 * (graph_nx.edges[edge]["score"] / max(max_edge, 1e-6)),
                        color=relation_colors.get(relation, "#c2410c"),
                        linestyle="solid" if relation == "next" else "dashed",
                        alpha=0.78,
                        connectionstyle=f"arc3,rad={rad}",
                        zorder=2,
                    )
                )

        node_scores = np.array([graph_nx.nodes[n]["score"] for n in graph_nx.nodes()], dtype=float)
        max_node = max(float(node_scores.max()), 1e-6) if len(node_scores) else 1.0
        for node_id, attrs in graph_nx.nodes(data=True):
            x, y = pos[node_id]
            score = float(attrs["score"])
            color = node_colors.get(attrs["node_type"], "#64748b")
            size = 260 + 560 * (score / max_node)
            ax_graph.scatter(
                [x],
                [y],
                s=size,
                color=color,
                alpha=0.9,
                edgecolors="white",
                linewidths=1.6,
                zorder=3,
            )
            label_color = color if attrs["node_type"] == "activity" and score >= max_node * 0.75 else "#0f172a"
            label_text = f"{wrapped_node_label(attrs)}\n{score:.2f}"
            ax_graph.text(
                x,
                y,
                label_text,
                ha="center",
                va="center",
                fontsize=8.4,
                color=label_color,
                weight="bold" if score >= max_node * 0.75 else "normal",
                linespacing=0.95,
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "alpha": 0.96},
                zorder=4,
            )

        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Case context" if node_type == "trace" else node_type.title(),
                markerfacecolor=color,
                markersize=9,
            )
            for node_type, color in node_colors.items()
            if any(graph_nx.nodes[n]["node_type"] == node_type for n in graph_nx.nodes())
        ]
        legend_handles.extend(
            [
                plt.Line2D([0], [0], color="#06b6d4", lw=2.5, label="Sequence link"),
                plt.Line2D([0], [0], color="#94a3b8", lw=2.0, linestyle="--", label="Case-context link"),
            ]
        )
        ax_graph.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=True,
            ncol=min(6, len(legend_handles)),
            borderaxespad=0.2,
        )

        last_event_x = float(x_positions[-1])
        if prediction["task"] == "activity":
            ax_graph.text(
                output_x,
                4.1,
                "Model Output",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="#475569",
            )
            ax_graph.annotate(
                "",
                xy=(output_x - 0.45, row_y["activity"] + 0.27),
                xytext=(last_event_x + 0.45, row_y["activity"]),
                arrowprops=dict(arrowstyle="->", color="#1d4ed8", linewidth=2.0, linestyle="--"),
                zorder=2,
            )
            ax_graph.annotate(
                "",
                xy=(output_x - 0.45, row_y["activity"] - 0.27),
                xytext=(last_event_x + 0.45, row_y["activity"]),
                arrowprops=dict(arrowstyle="->", color="#64748b", linewidth=2.0, linestyle=":"),
                zorder=2,
            )
            ax_graph.text(
                output_x,
                row_y["activity"] + 0.42,
                textwrap.fill(f"Predicted next: {prediction['predicted_label']}", width=26),
                ha="center",
                va="center",
                fontsize=9,
                color="#0f172a",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.24", "fc": "#dbeafe", "ec": "#1d4ed8", "alpha": 0.98},
                zorder=5,
            )
            ax_graph.text(
                output_x,
                row_y["activity"] - 0.28,
                textwrap.fill(f"Ground truth: {prediction['true_label']}", width=26),
                ha="center",
                va="center",
                fontsize=9,
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.24", "fc": "#f8fafc", "ec": "#64748b", "alpha": 0.98},
                zorder=5,
            )
            ax_graph.text(
                output_x,
                row_y["activity"] - 0.78,
                f"Confidence: {prediction['confidence']:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#475569",
                zorder=5,
            )
        else:
            ax_graph.text(
                output_x,
                4.1,
                "Model Output",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="#475569",
            )
            ax_graph.annotate(
                "",
                xy=(output_x - 0.45, row_y["time"] + 0.28),
                xytext=(last_event_x + 0.45, row_y["time"]),
                arrowprops=dict(arrowstyle="->", color="#1d4ed8", linewidth=2.0, linestyle="--"),
                zorder=2,
            )
            ax_graph.text(
                output_x,
                row_y["time"] + 0.45,
                textwrap.fill(f"Predicted: {prediction['prediction_display']}", width=24),
                ha="center",
                va="center",
                fontsize=9,
                color="#0f172a",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.24", "fc": "#dbeafe", "ec": "#1d4ed8", "alpha": 0.98},
                zorder=5,
            )
            ax_graph.text(
                output_x,
                row_y["time"] - 0.25,
                textwrap.fill(f"Actual: {prediction['target_display']}", width=24),
                ha="center",
                va="center",
                fontsize=9,
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.24", "fc": "#f8fafc", "ec": "#64748b", "alpha": 0.98},
                zorder=5,
            )

        title_lines = self._format_prediction_summary_lines(prediction)
        wrapped_title_lines = []
        case_title = self._case_title_line(prediction)
        if case_title:
            wrapped_title_lines.append(case_title)
        for line in title_lines:
            if ": " in line:
                label, value = line.split(": ", 1)
                wrapped = textwrap.wrap(
                    value,
                    width=95,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if wrapped:
                    wrapped_title_lines.append(f"{label}: {wrapped[0]}")
                    wrapped_title_lines.extend([f"    {part}" for part in wrapped[1:]])
                else:
                    wrapped_title_lines.append(f"{label}:")
            else:
                wrapped_title_lines.extend(
                    textwrap.wrap(
                        line,
                        width=105,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )

        fig.text(
            0.5,
            0.985,
            "Local Explanation",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="#0f172a",
        )
        fig.text(
            0.5,
            0.962,
            "\n".join(wrapped_title_lines),
            ha="center",
            va="top",
            fontsize=11,
            color="#0f172a",
        )
        fig.subplots_adjust(top=0.82)
        ax_graph.set_xlim(0.0, x_max)
        ax_graph.set_ylim(-0.05, 4.35)
        ax_graph.set_axis_off()

        ax_text.axis("off")

        top_nodes = node_df.sort_values("score", ascending=False).head(3) if not node_df.empty else pd.DataFrame()
        top_edges = edge_df.sort_values("score", ascending=False).head(2) if not edge_df.empty else pd.DataFrame()

        if prediction["task"] == "activity":
            is_correct = prediction["predicted_label"] == prediction["true_label"]
            status_label = "Correct prediction" if is_correct else "Incorrect prediction"
            status_color = "#15803d" if is_correct else "#b91c1c"
            confidence = float(prediction["confidence"])
            confidence_label = (
                "High confidence" if confidence >= 0.7 else
                "Moderate confidence" if confidence >= 0.4 else
                "Low confidence"
            )
            cards = [
                (
                    "Prediction status",
                    status_label,
                    f"Predicted: {prediction['predicted_label']}",
                    "#f8fafc",
                    status_color,
                ),
                (
                    "Ground truth",
                    prediction["true_label"],
                    "Observed next event",
                    "#f8fafc",
                    "#475569",
                ),
                (
                    "Model confidence",
                    f"{confidence:.3f}",
                    confidence_label,
                    "#eff6ff",
                    "#1d4ed8",
                ),
            ]
        else:
            cards = [
                (
                    "Predicted",
                    prediction["prediction_display"],
                    "Model output",
                    "#eff6ff",
                    "#1d4ed8",
                ),
                (
                    "Actual",
                    prediction["target_display"],
                    "Observed value",
                    "#f8fafc",
                    "#475569",
                ),
                (
                    "Question",
                    "Main drivers",
                    prediction.get("explanation_question", self._regression_question()),
                    "#f8fafc",
                    "#0f172a",
                ),
            ]

        for idx, (title, value, subtitle, fill_color, accent_color) in enumerate(cards):
            left = 0.02 + idx * 0.32
            ax_text.text(
                left,
                0.88,
                title,
                ha="left",
                va="top",
                fontsize=9,
                color="#475569",
                weight="bold",
                transform=ax_text.transAxes,
            )
            ax_text.text(
                left,
                0.68,
                textwrap.fill(str(value), width=26, break_long_words=False, break_on_hyphens=False),
                ha="left",
                va="top",
                fontsize=12,
                color=accent_color,
                weight="bold",
                bbox={"boxstyle": "round,pad=0.35", "fc": fill_color, "ec": accent_color, "alpha": 0.98},
                transform=ax_text.transAxes,
            )
            ax_text.text(
                left,
                0.38,
                textwrap.fill(str(subtitle), width=28, break_long_words=False, break_on_hyphens=False),
                ha="left",
                va="top",
                fontsize=8.5,
                color="#64748b",
                transform=ax_text.transAxes,
            )

        if not top_nodes.empty:
            driver_parts = [
                f"{row.label} ({row.score:.2f})"
                for row in top_nodes.itertuples()
            ]
            driver_text = textwrap.fill(
                "Top drivers: " + " | ".join(driver_parts),
                width=155,
                break_long_words=False,
                break_on_hyphens=False,
            )
            ax_text.text(
                0.02,
                0.16,
                driver_text,
                ha="left",
                va="center",
                fontsize=9,
                color="#0f172a",
                transform=ax_text.transAxes,
            )

        if not top_edges.empty:
            relation_parts = [
                f"{row.source_label} -> {row.target_label}"
                for row in top_edges.itertuples()
            ]
            relation_text = textwrap.fill(
                "Strongest links: " + " | ".join(relation_parts),
                width=155,
                break_long_words=False,
                break_on_hyphens=False,
            )
            ax_text.text(
                0.02,
                0.02,
                relation_text,
                ha="left",
                va="bottom",
                fontsize=8.5,
                color="#475569",
                transform=ax_text.transAxes,
            )

        ax_text.text(
            0.5,
            0.0,
            "Numerical values represent post-hoc explanation weights for nodes and edges in the explanation graph.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#475569",
            transform=ax_text.transAxes,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#f8fafc", "ec": "#cbd5e1", "alpha": 1.0},
        )

        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    def global_view_importance(self, graphs):
        rows = []
        view_rows = []
        fixed_views = ["time", "resource", "activity", "trace"]
        has_informative_resource = self._has_informative_resource_view()
        for graph in tqdm(graphs, desc="Global explainability"):
            explanation, _ = self.explain_graph(graph)
            graph_view_scores = {view: 0.0 for view in fixed_views}
            for node_type in fixed_views:
                if node_type == "resource" and not has_informative_resource:
                    continue
                if node_type not in graph.node_types:
                    continue
                node_mask = getattr(explanation[node_type], "node_mask", None)
                if node_mask is None:
                    continue
                node_mask = node_mask.detach().cpu().numpy()
                node_scores = np.abs(node_mask).mean(axis=1) if node_mask.ndim > 1 else np.abs(node_mask)
                graph_view_scores[node_type] = float(np.mean(node_scores)) if len(node_scores) else 0.0
            for view, score in graph_view_scores.items():
                view_rows.append({"view": view, "score": score})

            node_mask = getattr(explanation["activity"], "node_mask", None)
            if node_mask is None:
                continue
            node_mask = node_mask.detach().cpu().numpy()
            node_scores = np.abs(node_mask).mean(axis=1) if node_mask.ndim > 1 else np.abs(node_mask)
            for idx, score in enumerate(node_scores):
                label = self._node_label(graph, "activity", idx)
                rows.append({"activity": label, "score": float(score)})

        self._last_global_view_type_importance = self._summarize_view_type_importance(view_rows)
        if not rows:
            return pd.DataFrame(columns=["activity", "mean_score", "max_score", "count", "rank"])

        df = pd.DataFrame(rows)
        summary = (
            df.groupby("activity", as_index=False)
            .agg(
                mean_score=("score", "mean"),
                max_score=("score", "max"),
                count=("score", "size"),
            )
            .sort_values(["mean_score", "max_score"], ascending=False)
            .reset_index(drop=True)
        )
        summary["rank"] = np.arange(1, len(summary) + 1)
        return summary

    @staticmethod
    def _summarize_view_type_importance(view_rows):
        fixed_views = ["time", "resource", "activity", "trace"]
        if view_rows:
            raw_df = pd.DataFrame(view_rows)
        else:
            raw_df = pd.DataFrame(columns=["view", "score"])

        rows = []
        for rank, view in enumerate(fixed_views, start=1):
            scores = raw_df[raw_df["view"] == view]["score"].to_numpy(dtype=float) if not raw_df.empty else np.array([])
            if len(scores):
                mean_score = float(np.mean(scores))
                max_score = float(np.max(scores))
                count = int(np.count_nonzero(scores))
            else:
                mean_score = 0.0
                max_score = 0.0
                count = 0
            rows.append(
                {
                    "view": view,
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "count": count,
                    "rank": rank,
                }
            )
        return pd.DataFrame(rows)

    def save_local_artifacts(self, artifacts, output_dir, sample_id):
        os.makedirs(output_dir, exist_ok=True)

        node_df = pd.DataFrame(artifacts.node_rows)
        edge_df = pd.DataFrame(artifacts.edge_rows)
        if not node_df.empty:
            node_df = node_df.sort_values("score", ascending=False)
        if not edge_df.empty:
            edge_df = edge_df.sort_values("score", ascending=False)

        node_path = os.path.join(output_dir, f"sample_{sample_id}_nodes.csv")
        edge_path = os.path.join(output_dir, f"sample_{sample_id}_edges.csv")
        pred_path = os.path.join(output_dir, f"sample_{sample_id}_summary.json")
        desc_path = os.path.join(output_dir, f"sample_{sample_id}_description.txt")

        node_df.to_csv(node_path, index=False)
        edge_df.to_csv(edge_path, index=False)
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(artifacts.prediction_summary, handle, indent=2, default=_serialize_scalar)
        with open(desc_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self._sample_description(artifacts)))

        _remove_file_if_exists(os.path.join(output_dir, f"sample_{sample_id}_explanation.png"))
        _remove_file_if_exists(os.path.join(output_dir, f"sample_{sample_id}_graphviz_process.png"))
        self._draw_local_hetero_graph(
            artifacts,
            os.path.join(output_dir, f"sample_{sample_id}_hetero_graph.png"),
        )

    def save_global_artifacts(self, global_df, output_dir, num_graphs_used):
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "global_view_importance.csv")
        global_df.to_csv(csv_path, index=False)
        global_df.to_csv(os.path.join(output_dir, "global_activity_importance.csv"), index=False)
        _remove_file_if_exists(os.path.join(output_dir, "prophet_global_activity_importance.csv"))
        _remove_file_if_exists(os.path.join(output_dir, "prophet_global_activity_importance.png"))
        view_type_df = getattr(
            self,
            "_last_global_view_type_importance",
            self._summarize_view_type_importance([]),
        ).copy()
        view_type_df.to_csv(os.path.join(output_dir, "global_node_type_view_importance.csv"), index=False)
        self._plot_global_view_type_importance(
            view_type_df,
            os.path.join(output_dir, "global_node_type_view_importance.png"),
        )

        top_df = global_df.head(20).copy()
        fig_height = max(6, 0.4 * max(len(top_df), 1) + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(top_df["activity"][::-1], top_df["mean_score"][::-1], color="#1d4e89")
        ax.set_title(f"Global Activity Importance (n={int(num_graphs_used)} graphs)")
        ax.set_xlabel("Mean Explanation Score")
        ax.set_ylabel("Activity")
        for row_idx, row in enumerate(top_df.iloc[::-1].itertuples(), start=0):
            ax.text(
                row.mean_score + max(top_df["mean_score"].max(), 1e-6) * 0.01,
                row_idx,
                f"count={int(row.count)}",
                va="center",
                fontsize=8,
                color="#475569",
            )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "global_activity_importance.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_global_view_type_importance(view_type_df, output_path):
        fixed_views = ["time", "resource", "activity", "trace"]
        plot_df = (
            view_type_df.set_index("view")
            .reindex(fixed_views, fill_value=0.0)
            .reset_index()
        )
        scores = plot_df["mean_score"].to_numpy(dtype=float)
        y_max = max(0.15, float(scores.max()) * 1.12 if len(scores) else 0.15)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(plot_df["view"], scores, color="#2ca02c")
        ax.set_title("Global View Importance", fontsize=20, pad=12)
        ax.set_xlabel("View", fontsize=14)
        ax.set_ylabel("Mean Explanation Score", fontsize=14)
        ax.set_ylim(0.0, y_max)
        ax.grid(True, axis="both", color="#cfcfcf", linewidth=1.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=12)
        for spine in ax.spines.values():
            spine.set_color("#cfcfcf")
            spine.set_linewidth(1.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)


class GradientExplainer(ProphetGNNExplainer):
    pass


class TemporalGradientExplainer(ProphetGNNExplainer):
    pass


class GraphLIMEExplainer(ProphetGNNExplainer):
    pass


class GNNExplainabilityBenchmark:
    """
    Lightweight benchmark for heterogeneous GNN explanations.

    The benchmark evaluates activity-node attributions because they are the most
    stable, comparable sequence-aligned signal across graphs and tasks.
    """

    def __init__(self, explainer, task="activity"):
        self.explainer = explainer
        self.task = task
        self.results = {}

    @staticmethod
    def _rankdata(values):
        return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()

    @classmethod
    def _pearson_corr(cls, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2 or y.size < 2:
            return None
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return None
        corr = np.corrcoef(x, y)[0, 1]
        return None if not np.isfinite(corr) else float(corr)

    @classmethod
    def _spearman_corr(cls, x, y):
        return cls._pearson_corr(cls._rankdata(x), cls._rankdata(y))

    def _predict_graph(self, graph):
        graph = graph.to(self.explainer.device)
        self.explainer.model.eval()
        with torch.no_grad():
            act, event_time, remaining_time = self.explainer.model(graph)
            if self.task == "activity":
                probs = torch.softmax(act, dim=-1).view(-1)
                pred_idx = int(torch.argmax(probs).item())
                return {
                    "raw": probs.detach().cpu().numpy(),
                    "target_idx": pred_idx,
                    "score": float(probs[pred_idx].item()),
                }
            if self.task == "event_time":
                value = float(event_time.view(-1)[0].item())
                return {"raw": value, "target_idx": None, "score": value}
            value = float(remaining_time.view(-1)[0].item())
            return {"raw": value, "target_idx": None, "score": value}

    def _prediction_change(self, original_pred, updated_pred):
        if self.task == "activity":
            target_idx = original_pred["target_idx"]
            return float(original_pred["raw"][target_idx] - updated_pred["raw"][target_idx])
        return float(abs(original_pred["score"] - updated_pred["score"]))

    def _mask_activity_positions(self, graph, indices, keep_only=False):
        graph = graph.clone()
        x = graph["activity"].x.clone()
        indices = np.asarray(indices, dtype=np.int64).copy()
        if keep_only:
            masked = torch.zeros_like(x)
            if len(indices):
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=x.device)
                masked[index_tensor] = x[index_tensor]
            graph["activity"].x = masked
        else:
            if len(indices):
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=x.device)
                x[index_tensor] = 0
            graph["activity"].x = x
        return graph

    @staticmethod
    def _top_k_indices(sample_attr, k):
        sample_attr = np.asarray(sample_attr, dtype=float).reshape(-1)
        if sample_attr.size == 0:
            return np.array([], dtype=int)
        k = min(int(k), sample_attr.size)
        if k <= 0:
            return np.array([], dtype=int)
        return np.argsort(np.abs(sample_attr))[-k:]

    def _extract_activity_attributions(self, graph):
        explanation, _ = self.explainer.explain_graph(graph)
        node_mask = getattr(explanation["activity"], "node_mask", None)
        if node_mask is None:
            return None
        node_mask = node_mask.detach().cpu().numpy()
        return np.abs(node_mask).mean(axis=1) if node_mask.ndim > 1 else np.abs(node_mask)

    def _extract_benchmark_attributions(self, graph):
        explanation, _ = self.explainer.explain_graph(graph)
        node_mask = getattr(explanation["activity"], "node_mask", None)
        if node_mask is None:
            return None, None

        node_mask = node_mask.detach().cpu().numpy()
        node_attr = np.abs(node_mask).mean(axis=1) if node_mask.ndim > 1 else np.abs(node_mask)
        edge_endpoint_attr = np.zeros_like(node_attr, dtype=float)

        explanation_edge_types = getattr(explanation, "edge_types", [])
        for edge_type in explanation_edge_types:
            edge_mask = getattr(explanation[edge_type], "edge_mask", None)
            if edge_mask is None:
                continue
            edge_scores = np.abs(edge_mask.detach().cpu().numpy().reshape(-1))
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            source_type, _, target_type = edge_type
            for edge_idx, score in enumerate(edge_scores):
                src_idx = int(edge_index[0, edge_idx])
                dst_idx = int(edge_index[1, edge_idx])
                if source_type == "activity" and src_idx < len(edge_endpoint_attr):
                    edge_endpoint_attr[src_idx] += float(score)
                if target_type == "activity" and dst_idx < len(edge_endpoint_attr):
                    edge_endpoint_attr[dst_idx] += float(score)

        return node_attr, edge_endpoint_attr

    def _collect_samples(self, graphs, max_samples=10):
        samples = []
        for graph in tqdm(graphs[:max_samples], desc="Collecting benchmark samples"):
            attr, edge_attr = self._extract_benchmark_attributions(graph)
            if attr is None or len(attr) == 0:
                continue
            samples.append(
                {
                    "graph": graph,
                    "attr": np.asarray(attr, dtype=float),
                    "edge_attr": None if edge_attr is None else np.asarray(edge_attr, dtype=float),
                }
            )
        return samples

    def faithfulness_correlation(self, samples, k_values=(5, 10, 15, 20, 25)):
        results = {}
        for k in k_values:
            pred_changes = []
            importance_sums = []
            for sample in samples:
                attr = sample["attr"]
                if len(attr) < k:
                    continue
                top_k = self._top_k_indices(attr, k)
                if top_k.size == 0:
                    continue
                original_pred = self._predict_graph(sample["graph"])
                masked_graph = self._mask_activity_positions(sample["graph"], top_k, keep_only=False)
                masked_pred = self._predict_graph(masked_graph)
                pred_changes.append(self._prediction_change(original_pred, masked_pred))
                importance_sums.append(float(np.abs(attr[top_k]).sum()))
            valid_count = len(pred_changes)
            results[f"faithfulness_k{k}"] = {
                "spearman_correlation": self._spearman_corr(importance_sums, pred_changes),
                "pearson_correlation": self._pearson_corr(importance_sums, pred_changes),
                "mean_pred_change": float(np.mean(pred_changes)) if pred_changes else None,
                "std_pred_change": float(np.std(pred_changes)) if pred_changes else None,
                "valid_sample_count": valid_count,
            }
        return results

    def comprehensiveness(self, samples, k_values=(5, 10, 15, 20, 25)):
        results = {}
        for k in k_values:
            scores = []
            for sample in samples:
                attr = sample["attr"]
                if len(attr) < k:
                    continue
                top_k = self._top_k_indices(attr, k)
                if top_k.size == 0:
                    continue
                original_pred = self._predict_graph(sample["graph"])
                masked_graph = self._mask_activity_positions(sample["graph"], top_k, keep_only=False)
                masked_pred = self._predict_graph(masked_graph)
                scores.append(self._prediction_change(original_pred, masked_pred))
            valid_count = len(scores)
            results[f"comprehensiveness_k{k}"] = {
                "mean": float(np.mean(scores)) if scores else None,
                "std": float(np.std(scores)) if scores else None,
                "median": float(np.median(scores)) if scores else None,
                "valid_sample_count": valid_count,
            }
        return results

    def sufficiency(self, samples, k_values=(5, 10, 15, 20, 25)):
        results = {}
        for k in k_values:
            scores = []
            for sample in samples:
                attr = sample["attr"]
                if len(attr) < k:
                    continue
                top_k = self._top_k_indices(attr, k)
                if top_k.size == 0:
                    continue
                original_pred = self._predict_graph(sample["graph"])
                top_only_graph = self._mask_activity_positions(sample["graph"], top_k, keep_only=True)
                top_only_pred = self._predict_graph(top_only_graph)
                scores.append(self._prediction_change(original_pred, top_only_pred))
            valid_count = len(scores)
            results[f"sufficiency_k{k}"] = {
                "mean": float(np.mean(scores)) if scores else None,
                "std": float(np.std(scores)) if scores else None,
                "median": float(np.median(scores)) if scores else None,
                "valid_sample_count": valid_count,
            }
        return results

    def agreement(self, samples, k_values=(5, 10, 15, 20, 25)):
        """
        GNN-specific agreement between two explanation views:
        activity-node attributions and activity-node scores induced by incident
        edge attributions. This keeps agreement meaningful without requiring a
        second external explainer.
        """
        results = {}
        for k in k_values:
            jaccard_scores = []
            overlap_scores = []
            for sample in samples:
                node_attr = np.asarray(sample["attr"], dtype=float).reshape(-1)
                edge_attr = sample.get("edge_attr")
                if edge_attr is None:
                    continue
                edge_attr = np.asarray(edge_attr, dtype=float).reshape(-1)
                usable_len = min(len(node_attr), len(edge_attr))
                if usable_len < k:
                    continue
                node_top = set(self._top_k_indices(node_attr[:usable_len], k).tolist())
                edge_top = set(self._top_k_indices(edge_attr[:usable_len], k).tolist())
                if not node_top or not edge_top:
                    continue
                intersection = len(node_top.intersection(edge_top))
                union = len(node_top.union(edge_top))
                jaccard_scores.append(float(intersection / union) if union else None)
                overlap_scores.append(float(intersection / k))

            valid_count = len([score for score in jaccard_scores if score is not None])
            results[f"agreement_k{k}"] = {
                "jaccard_similarity": float(np.mean(jaccard_scores)) if valid_count else None,
                "top_k_overlap": float(np.mean(overlap_scores)) if overlap_scores else None,
                "valid_sample_count": valid_count,
            }
        return results

    def monotonicity(self, samples):
        monotonicity_scores = []
        for sample in samples[:8]:
            attr = sample["attr"]
            order = np.argsort(np.abs(attr))[::-1]
            if order.size <= 1:
                continue
            original_pred = self._predict_graph(sample["graph"])
            previous_score = original_pred["score"]
            monotonic_steps = 0
            for cut in range(1, min(len(order), 10) + 1):
                masked_graph = self._mask_activity_positions(sample["graph"], order[:cut], keep_only=False)
                masked_pred = self._predict_graph(masked_graph)
                current_score = masked_pred["score"]
                if self.task == "activity":
                    if current_score <= previous_score:
                        monotonic_steps += 1
                else:
                    if abs(current_score - original_pred["score"]) >= abs(previous_score - original_pred["score"]):
                        monotonic_steps += 1
                previous_score = current_score
            denom = max(min(len(order), 10), 1)
            monotonicity_scores.append(monotonic_steps / denom)
        return {
            "monotonicity": {
                "mean": float(np.mean(monotonicity_scores)) if monotonicity_scores else None,
                "std": float(np.std(monotonicity_scores)) if monotonicity_scores else None,
                "median": float(np.median(monotonicity_scores)) if monotonicity_scores else None,
                "valid_sample_count": len(monotonicity_scores),
            }
        }

    def stability(self, samples):
        cosine_scores = []
        variance_scores = []
        for sample in samples[:5]:
            attr = sample["attr"]
            recomputed = []
            for cut in range(min(3, len(attr))):
                mask_idx = np.array([int(np.argsort(np.abs(attr))[cut])], dtype=int)
                perturbed_graph = self._mask_activity_positions(sample["graph"], mask_idx, keep_only=False)
                perturbed_attr = self._extract_activity_attributions(perturbed_graph)
                if perturbed_attr is None:
                    continue
                min_len = min(len(attr), len(perturbed_attr))
                base = np.asarray(attr[:min_len], dtype=float)
                pert = np.asarray(perturbed_attr[:min_len], dtype=float)
                denom = np.linalg.norm(base) * np.linalg.norm(pert)
                cosine = 1.0 if denom == 0 and np.allclose(base, pert) else (float(np.dot(base, pert) / denom) if denom > 0 else 0.0)
                if np.isfinite(cosine):
                    cosine_scores.append(cosine)
                recomputed.append(pert)
            if recomputed:
                variance_scores.append(float(np.var(np.stack(recomputed, axis=0), axis=0).mean()))
        return {
            "stability": {
                "mean_variance": float(np.mean(variance_scores)) if variance_scores else 0.0,
                "max_variance": float(np.max(variance_scores)) if variance_scores else 0.0,
                "mean_cosine_similarity": float(np.mean(cosine_scores)) if cosine_scores else 0.0,
                "stability_score": float(np.mean(cosine_scores)) if cosine_scores else 0.0,
            }
        }

    @staticmethod
    def _hoyer_sparsity(values):
        values = np.abs(np.asarray(values, dtype=float).reshape(-1))
        n = values.size
        if n <= 1:
            return None
        l2 = float(np.linalg.norm(values))
        if l2 <= 0:
            return None
        l1 = float(values.sum())
        score = (np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _effective_active_fraction(values):
        values = np.abs(np.asarray(values, dtype=float).reshape(-1))
        n = values.size
        if n == 0:
            return None
        total = float(values.sum())
        sq_sum = float(np.square(values).sum())
        if total <= 0 or sq_sum <= 0:
            return None
        effective_nodes = (total * total) / sq_sum
        return float(np.clip(effective_nodes / n, 0.0, 1.0))

    def sparsity(self, samples):
        sparsity_scores = []
        active_fractions = []
        mass_top3 = []
        mass_top5 = []
        for sample in samples:
            attr = np.abs(np.asarray(sample["attr"], dtype=float).reshape(-1))
            if attr.size == 0:
                continue
            total_mass = float(attr.sum())
            if total_mass <= 0:
                continue
            sparsity_score = self._hoyer_sparsity(attr)
            active_fraction = self._effective_active_fraction(attr)
            if sparsity_score is not None:
                sparsity_scores.append(sparsity_score)
            if active_fraction is not None:
                active_fractions.append(active_fraction)
            sorted_mass = np.sort(attr)[::-1]
            mass_top3.append(float(sorted_mass[: min(3, len(sorted_mass))].sum() / total_mass))
            mass_top5.append(float(sorted_mass[: min(5, len(sorted_mass))].sum() / total_mass))
        return {
            "sparsity": {
                "active_fraction": float(np.mean(active_fractions)) if active_fractions else None,
                "sparsity_score": float(np.mean(sparsity_scores)) if sparsity_scores else None,
                "top3_mass_fraction": float(np.mean(mass_top3)) if mass_top3 else None,
                "top5_mass_fraction": float(np.mean(mass_top5)) if mass_top5 else None,
                "valid_sample_count": len(sparsity_scores),
            }
        }

    def temporal_consistency(self, samples):
        if not samples:
            return {
                "temporal_consistency": {
                    "recency_correlation": None,
                    "position_importance": [],
                    "valid_sample_count": 0,
                }
            }
        num_bins = 10
        totals = np.zeros(num_bins, dtype=float)
        counts = np.zeros(num_bins, dtype=float)
        recency_correlations = []
        for sample in samples:
            attr = np.abs(np.asarray(sample["attr"], dtype=float).reshape(-1))
            if attr.size <= 1:
                continue
            relative_position = np.linspace(0.0, 1.0, attr.size)
            corr = self._spearman_corr(relative_position, attr)
            if corr is not None:
                recency_correlations.append(float(corr))

            bin_indices = np.minimum((relative_position * num_bins).astype(int), num_bins - 1)
            for bin_idx, score in zip(bin_indices, attr):
                totals[bin_idx] += float(score)
                counts[bin_idx] += 1.0

        avg = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        valid = np.where(counts > 0)[0]
        position_importance = [
            float(avg[idx]) if counts[idx] > 0 else None
            for idx in range(num_bins)
        ]
        return {
            "temporal_consistency": {
                "recency_correlation": float(np.mean(recency_correlations)) if recency_correlations else None,
                "position_importance": position_importance,
                "most_important_position": int(valid[np.argmax(avg[valid])]) if len(valid) else 0,
                "least_important_position": int(valid[np.argmin(avg[valid])]) if len(valid) else 0,
                "valid_sample_count": len(recency_correlations),
            }
        }

    def run_full_benchmark(self, graphs, k_values=(5, 10, 15, 20, 25), max_samples=10):
        print("\n" + "=" * 60)
        print("GNN EXPLAINABILITY BENCHMARK EVALUATION")
        print("=" * 60)
        samples = self._collect_samples(graphs, max_samples=max_samples)
        results = {
            "metadata": {
                "task": self.task,
                "n_samples": len(samples),
                "k_values": list(k_values),
            }
        }
        results["faithfulness"] = self.faithfulness_correlation(samples, k_values=k_values)
        results["comprehensiveness"] = self.comprehensiveness(samples, k_values=k_values)
        results["sufficiency"] = self.sufficiency(samples, k_values=k_values)
        results["agreement"] = self.agreement(samples, k_values=k_values)
        results["monotonicity"] = self.monotonicity(samples)
        results["stability"] = self.stability(samples)
        results["sparsity"] = self.sparsity(samples)
        results["temporal_consistency"] = self.temporal_consistency(samples)
        self.results = results
        return results

    def save_results(self, output_dir, filename="benchmark_results.json"):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2, default=_serialize_scalar)
        print(f"[OK] Benchmark results saved to: {filepath}")

        summary_rows = []
        task_name = self.results.get("metadata", {}).get("task", self.task)
        for category, metric_data in self.results.items():
            if category == "metadata" or not isinstance(metric_data, dict):
                continue
            for sub_key, sub_val in metric_data.items():
                if isinstance(sub_val, dict):
                    for metric_name, value in sub_val.items():
                        if isinstance(value, (int, float)):
                            summary_rows.append(
                                {
                                    "category": category,
                                    "metric": f"{task_name}_{sub_key}_{metric_name}",
                                    "value": value,
                                }
                            )
                elif isinstance(sub_val, (int, float)):
                    summary_rows.append(
                        {
                            "category": category,
                            "metric": f"{task_name}_{sub_key}",
                            "value": sub_val,
                        }
                    )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "benchmark_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Benchmark summary saved to: {summary_path}")
        return filepath, summary_df


def _select_sample_indices(num_graphs, sample_count):
    sample_count = min(sample_count, num_graphs)
    if sample_count <= 0:
        return []
    if sample_count == 1:
        return [0]
    return sorted(set(np.linspace(0, num_graphs - 1, sample_count, dtype=int).tolist()))


def _collect_task_seconds(graphs, task):
    seconds_values = []
    for graph in graphs:
        if task == "event_time":
            raw = getattr(graph, "y_timestamp", None)
        elif task == "remaining_time":
            raw = getattr(graph, "y_remaining_time", None)
        else:
            continue
        if raw is None:
            continue
        raw_value = float(raw.view(-1)[0].item())
        seconds_values.append(ProphetGNNExplainer._decode_log_seconds(raw_value))
    return seconds_values


def _collect_display_seconds(graphs):
    seconds_values = []
    for graph in graphs:
        if "time" not in graph.node_types:
            continue
        display_seconds = getattr(graph["time"], "display_seconds", None)
        if display_seconds is None:
            continue
        values = display_seconds.detach().cpu().view(-1).numpy().tolist()
        seconds_values.extend(float(value) for value in values if np.isfinite(value) and float(value) >= 0)
    return seconds_values


def _to_optional_int(value):
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        if value == "":
            return None
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _graph_prefix_length(graph):
    raw_prefix = getattr(graph, "prefix_id", None)
    prefix_length = _to_optional_int(raw_prefix)
    if prefix_length is not None and prefix_length > 0:
        return prefix_length

    try:
        activity_store = graph["activity"]
        num_nodes = getattr(activity_store, "num_nodes", None)
        if num_nodes is not None:
            return int(num_nodes)
        x = getattr(activity_store, "x", None)
        if x is not None:
            return int(x.shape[0])
    except Exception:
        pass

    return None


def _filter_graphs_by_prefix_length(graphs, min_prefix_length=None, max_prefix_length=None):
    min_prefix = _to_optional_int(min_prefix_length)
    max_prefix = _to_optional_int(max_prefix_length)
    if min_prefix is None:
        min_prefix = 1
    if max_prefix is not None and max_prefix < min_prefix:
        raise RuntimeError(
            f"Invalid explainability prefix range: min_prefix_length={min_prefix}, max_prefix_length={max_prefix}."
        )

    filtered = []
    skipped_unknown = 0
    for graph in graphs:
        prefix_length = _graph_prefix_length(graph)
        if prefix_length is None:
            skipped_unknown += 1
            continue
        if prefix_length < min_prefix:
            continue
        if max_prefix is not None and prefix_length > max_prefix:
            continue
        filtered.append(graph)

    if not filtered:
        range_label = f">= {min_prefix}" if max_prefix is None else f"{min_prefix}..{max_prefix}"
        raise RuntimeError(
            "No test graphs matched the explainability prefix length range "
            f"({range_label}). Skipped {skipped_unknown} graphs with unknown prefix length."
        )

    return filtered, min_prefix, max_prefix


def run_gnn_explainability(
    model,
    data,
    output_dir,
    device,
    vocabularies=None,
    num_samples=50,
    local_num_samples=5,
    methods="all",
    tasks=None,
    scaler=None,
    y_true=None,
    run_benchmark=True,
    global_sample_percent=1,
    min_prefix_length=None,
    max_prefix_length=None,
):
    del methods, scaler, y_true
    os.makedirs(output_dir, exist_ok=True)

    graphs = data.get("test_graphs") or data.get("test") or []
    if not graphs:
        raise RuntimeError("No test graphs available for GNN explainability.")
    original_graph_count = len(graphs)
    graphs, applied_min_prefix, applied_max_prefix = _filter_graphs_by_prefix_length(
        graphs,
        min_prefix_length=min_prefix_length,
        max_prefix_length=max_prefix_length,
    )
    print(
        "[INFO] Explainability prefix filter: "
        f"min={applied_min_prefix}, max={applied_max_prefix or 'none'}, "
        f"kept={len(graphs)}/{original_graph_count} test graphs"
    )

    if tasks is None:
        tasks = ["activity"]
    elif isinstance(tasks, str):
        tasks = [tasks]

    summary = {}
    benchmark_results = {}
    benchmark_frames = []
    local_num_samples = max(0, int(local_num_samples))
    sample_indices = _select_sample_indices(len(graphs), local_num_samples)
    
    global_sample_percent = float(global_sample_percent)
    global_sample_percent = min(max(global_sample_percent, 0.0), 100.0)
    num_global_graphs = max(1, int(np.ceil(len(graphs) * (global_sample_percent / 100.0))))
    global_indices = _select_sample_indices(len(graphs), num_global_graphs)
    global_graphs = [graphs[i] for i in global_indices]

    for task in tasks:
        task_dir = os.path.join(output_dir, "prophet", task)
        local_dir = os.path.join(task_dir, "local")
        global_dir = os.path.join(task_dir, "global")
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(global_dir, exist_ok=True)

        display_seconds = _collect_display_seconds(graphs)
        preferred_time_unit = ProphetGNNExplainer.choose_preferred_time_unit(display_seconds)
        if task in {"event_time", "remaining_time"}:
            task_seconds = _collect_task_seconds(graphs, task)
            preferred_time_unit = ProphetGNNExplainer.choose_preferred_time_unit(task_seconds or display_seconds)

        explainer = ProphetGNNExplainer(
            model=model,
            device=device,
            task=task,
            vocabularies=vocabularies,
            preferred_time_unit=preferred_time_unit,
        )

        for sample_id, graph_idx in tqdm(enumerate(sample_indices), total=len(sample_indices), desc=f"Local explanations ({task})"):
            graph = graphs[graph_idx]
            explanation, prediction = explainer.explain_graph(graph)
            artifacts = explainer.summarize_local_explanation(graph, explanation, prediction)
            explainer.save_local_artifacts(artifacts, local_dir, sample_id)

        global_df = explainer.global_view_importance(global_graphs)
        explainer.save_global_artifacts(global_df, global_dir, len(global_graphs))

        if run_benchmark:
            benchmark_dir = os.path.join(output_dir, "benchmark")
            benchmark_graphs = list(global_graphs)
            benchmark = GNNExplainabilityBenchmark(explainer, task=task)
            task_benchmark_results = benchmark.run_full_benchmark(benchmark_graphs, max_samples=len(benchmark_graphs))
            task_filename = f"benchmark_results_{task}.json"
            _, task_summary_df = benchmark.save_results(benchmark_dir, filename=task_filename)
            benchmark_results[task] = task_benchmark_results
            if not task_summary_df.empty:
                task_summary_df = task_summary_df.copy()
                task_summary_df["task"] = task
                benchmark_frames.append(task_summary_df)

        summary[task] = {
            "sample_indices": sample_indices,
            "num_candidate_graphs": len(graphs),
            "original_num_candidate_graphs": original_graph_count,
            "min_prefix_length": applied_min_prefix,
            "max_prefix_length": applied_max_prefix,
            "num_global_graphs": len(global_graphs),
            "global_sample_percent": global_sample_percent,
            "top_global_view": None if global_df.empty else str(global_df.iloc[0]["activity"]),
            "top_global_activity": None if global_df.empty else str(global_df.iloc[0]["activity"]),
            "benchmark_enabled": bool(run_benchmark),
        }

    summary_path = os.path.join(output_dir, "prophet_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_serialize_scalar)

    if run_benchmark and benchmark_results:
        benchmark_dir = os.path.join(output_dir, "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        combined_json_path = os.path.join(benchmark_dir, "benchmark_results.json")
        with open(combined_json_path, "w", encoding="utf-8") as handle:
            json.dump(benchmark_results, handle, indent=2, default=_serialize_scalar)
        if benchmark_frames:
            combined_df = pd.concat(benchmark_frames, ignore_index=True)
            combined_df.to_csv(os.path.join(benchmark_dir, "benchmark_summary.csv"), index=False)
            print(f"[OK] Combined benchmark summary saved to: {os.path.join(benchmark_dir, 'benchmark_summary.csv')}")
        print(f"[OK] Combined benchmark results saved to: {combined_json_path}")

    prophet_root = os.path.join(output_dir, "prophet")
    if local_num_samples > 0 and not _dir_has_png(os.path.join(prophet_root, tasks[0], "local")):
        raise RuntimeError("PROPHET explainability failed: no local explanation plots were generated.")
    if not _dir_has_png(os.path.join(prophet_root, tasks[0], "global")):
        raise RuntimeError("PROPHET explainability failed: no global explanation plots were generated.")

    return summary


class GNNExplainerWrapper:
    def __init__(self, model, device, vocabularies=None, scaler=None):
        del scaler
        self.model = model
        self.device = device
        self.vocabularies = vocabularies

    def run(
        self,
        data,
        output_dir,
        num_samples=50,
        local_num_samples=5,
        methods="all",
        tasks=None,
        y_true=None,
        run_benchmark=True,
        global_sample_percent=1,
        min_prefix_length=None,
        max_prefix_length=None,
    ):
        return run_gnn_explainability(
            model=self.model,
            data=data,
            output_dir=output_dir,
            device=self.device,
            vocabularies=self.vocabularies,
            num_samples=num_samples,
            local_num_samples=local_num_samples,
            methods=methods,
            tasks=tasks,
            y_true=y_true,
            run_benchmark=run_benchmark,
            global_sample_percent=global_sample_percent,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
        )
