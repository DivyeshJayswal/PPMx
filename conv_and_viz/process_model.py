import os
from datetime import datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from conv_and_viz.xes_utils import normalized_xes_path


def load_event_log(xes_path: str):
    """Load an XES event log file."""
    with normalized_xes_path(xes_path) as import_path:
        return xes_importer.apply(import_path)


def ensure_output_folder(folder_path: str):
    """Create output folder if it doesn't exist."""
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def generate_dfg(event_log):
    """Generate frequency and performance DFG from event log."""
    dfg_freq = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.FREQUENCY)
    dfg_perf = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
    return dfg_freq, dfg_perf


def save_dfg_visual(dfg, log, output_folder: str, variant: str = "frequency", base_name: str = "dfg"):
    """
    Save DFG visualization as PNG image.
    
    Args:
        dfg: DFG object
        log: Event log
        output_folder: Output folder path
        variant: 'frequency' or 'performance'
        base_name: Base name for output file
        
    Returns:
        str: Path to saved image
    """
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{base_name}_dfg_{variant}_{ts}.png")

    if variant == "frequency":
        gviz = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.FREQUENCY)
    else:
        gviz = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.PERFORMANCE)

    dfg_visualizer.save(gviz, path)
    return path


def petri_from_log(event_log):
    """Discover Petri net from event log using Inductive Miner."""
    tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
    net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
    return net, im, fm


def save_petri_png(net, im, fm, output_folder: str, base_name: str = "petri"):
    """
    Save Petri net visualization as PNG image.
    
    Args:
        net: Petri net
        im: Initial marking
        fm: Final marking
        output_folder: Output folder path
        base_name: Base name for output file
        
    Returns:
        str: Path to saved image
    """
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{base_name}_petri_{ts}.png")
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz, path)
    return path


def build_and_save_all(xes_path: str, output_folder: str):
    """
    Build and save all process model visualizations.
    
    Args:
        xes_path: Path to XES event log file
        output_folder: Output folder for visualizations
        
    Returns:
        dict: Paths to generated files (frequency_jpg, performance_jpg, petri_jpg)
    """
    log = load_event_log(xes_path)
    
    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    
    dfg_freq, dfg_perf = generate_dfg(log)
    net, im, fm = petri_from_log(log)

    freq_path = save_dfg_visual(dfg_freq, log, output_folder, "frequency", base_name)
    perf_path = save_dfg_visual(dfg_perf, log, output_folder, "performance", base_name)
    petri_path = save_petri_png(net, im, fm, output_folder, base_name)

    return {
        "frequency_jpg": freq_path,
        "performance_jpg": perf_path,
        "petri_jpg": petri_path
    }


def build_and_save_dfg_visuals(xes_path: str, output_folder: str):
    """Build and save DFG visualizations only."""
    log = load_event_log(xes_path)
    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    dfg_freq, dfg_perf = generate_dfg(log)
    
    freq_path = save_dfg_visual(dfg_freq, log, output_folder, "frequency", base_name)
    perf_path = save_dfg_visual(dfg_perf, log, output_folder, "performance", base_name)
    
    return {
        "frequency_jpg": freq_path,
        "performance_jpg": perf_path
    }


def build_and_save_petri(xes_path: str, output_folder: str):
    """Build and save Petri net visualization only."""
    log = load_event_log(xes_path)
    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    net, im, fm = petri_from_log(log)
    
    petri_path = save_petri_png(net, im, fm, output_folder, base_name)
    
    return {
        "petri_jpg": petri_path
    }
