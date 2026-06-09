import pandas as pd
import numpy as np


class UniversalColumnDetector:
    KNOWN_PATTERNS = {
        'case_id': [
            'case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid',
            'Case ID', 'Case_ID', 'case', 'Case', 'trace_id', 'TraceID'
        ],
        'activity': [
            'concept:name', 'Activity', 'activity', 'event', 'Event',
            'task', 'Task', 'activity_name', 'ActivityName', 'Action'
        ],
        'timestamp': [
            'time:timestamp', 'Timestamp', 'timestamp', 'time', 'Time',
            'start_time', 'StartTime', 'event_time', 'EventTime',
            'complete_time', 'CompleteTime'
        ],
        'resource': [
            'org:resource', 'Resource', 'resource', 'user', 'User',
            'org:role', 'org:group', 'role', 'Role', 'actor', 'Actor',
            'agent', 'Agent'
        ]
    }
    
    def __init__(self, df):
        self.df = df
        self.detected_columns = {}
        self.column_mapping = {}
        
    def detect_all(self):
        self._detect_case_id()
        self._detect_activity()
        self._detect_timestamp()
        self._detect_resource()
        
        self._create_mapping()
        
        return self.detected_columns, self.column_mapping
    
    def _detect_case_id(self):
        for col in self.df.columns:
            if col in self.KNOWN_PATTERNS['case_id']:
                self.detected_columns['case_id'] = col
                return
        
        for col in self.df.columns:
            col_lower = col.lower()
            if 'case' in col_lower and ('id' in col_lower or 'name' in col_lower):
                self.detected_columns['case_id'] = col
                return
        
        for col in self.df.columns:
            if self._is_likely_case_id(col):
                self.detected_columns['case_id'] = col
                return
    
    def _detect_activity(self):
        for col in self.df.columns:
            if col in self.KNOWN_PATTERNS['activity']:
                self.detected_columns['activity'] = col
                return
        
        for col in self.df.columns:
            col_lower = col.lower()
            if 'activity' in col_lower or 'event' in col_lower or 'task' in col_lower or 'action' in col_lower:
                if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                    self.detected_columns['activity'] = col
                    return
    
    def _detect_timestamp(self):
        for col in self.df.columns:
            if col in self.KNOWN_PATTERNS['timestamp']:
                self.detected_columns['timestamp'] = col
                return
        
        for col in self.df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'date' in col_lower:
                try:
                    pd.to_datetime(self.df[col])
                    self.detected_columns['timestamp'] = col
                    return
                except:
                    continue
    
    def _detect_resource(self):
        for col in self.df.columns:
            if col in self.KNOWN_PATTERNS['resource']:
                self.detected_columns['resource'] = col
                return
        
        for col in self.df.columns:
            col_lower = col.lower()
            if 'resource' in col_lower or 'user' in col_lower or 'role' in col_lower or 'actor' in col_lower:
                if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                    self.detected_columns['resource'] = col
                    return
    
    def _is_likely_case_id(self, col):
        unique_ratio = self.df[col].nunique() / len(self.df)
        
        if unique_ratio > 0.5:
            return False
        
        if unique_ratio < 0.01:
            return False
        
        grouped = self.df.groupby(col).size()
        if grouped.mean() > 2:
            return True
        
        return False
    
    def _create_mapping(self):
        standard_names = {
            'case_id': 'CaseID',
            'activity': 'Activity',
            'timestamp': 'Timestamp',
            'resource': 'Resource'
        }
        
        for key, detected_col in self.detected_columns.items():
            standard_name = standard_names[key]
            if detected_col != standard_name:
                self.column_mapping[detected_col] = standard_name
    
    def apply_mapping(self):
        if self.column_mapping:
            self.df = self.df.rename(columns=self.column_mapping)
        return self.df
    
    def get_detection_report(self):
        report = []
        report.append("="*70)
        report.append("COLUMN DETECTION REPORT")
        report.append("="*70)
        
        for key, col in self.detected_columns.items():
            standard = self.column_mapping.get(col, col)
            if col in self.column_mapping:
                report.append(f"{key.upper()}: '{col}' → '{standard}'")
            else:
                report.append(f"{key.upper()}: '{col}' (already standard)")
        
        if not self.detected_columns:
            report.append("WARNING: No columns could be auto-detected")
            report.append("Please verify your dataset has: CaseID, Activity, Timestamp")
        
        report.append("="*70)
        
        return "\n".join(report)


def detect_and_standardize_columns(df, verbose=True):
    detector = UniversalColumnDetector(df)
    detected, mapping = detector.detect_all()
    
    if verbose:
        print(detector.get_detection_report())
    
    standardized_df = detector.apply_mapping()
    
    required = ['CaseID', 'Activity', 'Timestamp']
    missing = [col for col in required if col not in standardized_df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")
    
    return standardized_df, mapping, detected


def add_custom_pattern(category, patterns):
    if category in UniversalColumnDetector.KNOWN_PATTERNS:
        for pattern in patterns:
            if pattern not in UniversalColumnDetector.KNOWN_PATTERNS[category]:
                UniversalColumnDetector.KNOWN_PATTERNS[category].append(pattern)
    else:
        UniversalColumnDetector.KNOWN_PATTERNS[category] = patterns
