#!/usr/bin/env python3
"""
JSONL Reviewer - Review and edit JSON entries from a JSONL file

Opens a JSONL file, displays entries one at a time in a dark-themed PyQt6 window,
allows editing, and saves accepted entries to an output file.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class JSONLReviewer(QWidget):
    def __init__(self):
        super().__init__()
        self.input_file: Optional[Path] = None
        self.output_file: Optional[Path] = None
        self.entries = []
        self.current_index = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("JSONL Reviewer")
        self.setStyleSheet("background-color: #2E2E2E; color: ivory;")
        
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("No file loaded")
        self.status_label.setStyleSheet("color: ivory; padding: 5px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # JSON display/edit area
        self.text_edit = QTextEdit()
        self.text_edit.setStyleSheet(
            "background-color: #1E1E1E; "
            "color: ivory; "
            "border: 1px solid #555; "
            "padding: 10px;"
        )
        self.text_edit.setFont(QFont("Consolas", 14))
        self.text_edit.setPlaceholderText("No entry loaded. Click 'Select Input File' to begin.")
        layout.addWidget(self.text_edit)
        
        # Error label (for JSON validation errors)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #FF6B6B; padding: 5px; font-style: italic;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # File selection button
        self.select_file_button = QPushButton("Select Input File")
        self.select_file_button.setStyleSheet(
            "background-color: #3E3E3E; "
            "color: ivory; "
            "padding: 8px; "
            "border: 1px solid #555;"
        )
        self.select_file_button.clicked.connect(self.select_input_file)
        button_layout.addWidget(self.select_file_button)
        
        # Accept button
        self.accept_button = QPushButton("Accept")
        self.accept_button.setStyleSheet(
            "background-color: #4CAF50; "
            "color: white; "
            "padding: 8px; "
            "border: 1px solid #555; "
            "font-weight: bold;"
        )
        self.accept_button.clicked.connect(self.accept_entry)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)
        
        # Reject button
        self.reject_button = QPushButton("Reject")
        self.reject_button.setStyleSheet(
            "background-color: #F44336; "
            "color: white; "
            "padding: 8px; "
            "border: 1px solid #555; "
            "font-weight: bold;"
        )
        self.reject_button.clicked.connect(self.reject_entry)
        self.reject_button.setEnabled(False)
        button_layout.addWidget(self.reject_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(900, 700)
        
    def select_input_file(self):
        """Open file dialog to select input JSONL file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSONL File",
            "",
            "JSONL Files (*.jsonl);;All Files (*)"
        )
        
        if file_path:
            self.input_file = Path(file_path)
            self.output_file = self.input_file.parent / f"{self.input_file.stem}_reviewed.jsonl"
            self.load_entries()
            
    def load_entries(self):
        """Load all entries from the JSONL file."""
        self.entries = []
        self.current_index = 0
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        self.entries.append({
                            'line_num': line_num,
                            'original': line,
                            'data': entry
                        })
                    except json.JSONDecodeError as e:
                        # Store invalid JSON entries too
                        self.entries.append({
                            'line_num': line_num,
                            'original': line,
                            'data': None,
                            'error': str(e)
                        })
            
            if not self.entries:
                QMessageBox.warning(self, "Warning", "No valid entries found in file.")
                return
            
            self.status_label.setText(
                f"Loaded {len(self.entries)} entries from {self.input_file.name}"
            )
            self.display_current_entry()
            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {str(e)}")
            
    def display_current_entry(self):
        """Display the current entry in the text editor."""
        if not self.entries or self.current_index >= len(self.entries):
            return
        
        entry = self.entries[self.current_index]
        
        # Update status
        self.status_label.setText(
            f"Entry {self.current_index + 1} of {len(self.entries)} "
            f"(Line {entry['line_num']})"
        )
        
        # Format and display JSON
        if entry['data'] is not None:
            # Valid JSON - format it nicely
            formatted = json.dumps(entry['data'], indent=2, ensure_ascii=False)
            self.text_edit.setPlainText(formatted)
            self.error_label.setText("")
        else:
            # Invalid JSON - show original text with error
            self.text_edit.setPlainText(entry['original'])
            self.error_label.setText(
                f"⚠ JSON Error: {entry.get('error', 'Unknown error')}. "
                "Please repair the JSON before accepting."
            )
            
    def validate_json(self, text: str) -> tuple[bool, Optional[dict], Optional[str]]:
        """Validate JSON text. Returns (is_valid, parsed_data, error_message)."""
        try:
            data = json.loads(text.strip())
            return True, data, None
        except json.JSONDecodeError as e:
            error_msg = str(e)
            # Provide more helpful error message for common issues
            if "Expecting ',' delimiter" in error_msg or "Unterminated string" in error_msg:
                # Check for common quote/escape issues
                if '\\""' in text or '\\"' in text:
                    error_msg += "\n\nTip: Check for double backslashes (\\\\) before quotes. Use single backslash (\\\") for quotes inside strings."
            return False, None, error_msg
            
    def accept_entry(self):
        """Accept the current entry and save to output file."""
        text = self.text_edit.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Warning", "Entry is empty. Cannot accept.")
            return
        
        # Validate JSON
        is_valid, data, error = self.validate_json(text)
        
        if not is_valid:
            reply = QMessageBox.question(
                self,
                "Invalid JSON",
                f"The entry contains invalid JSON:\n\n{error}\n\n"
                "Do you want to accept it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            # Accept invalid JSON as-is (user's choice)
            json_line = text
        else:
            # Valid JSON - save as compact JSONL line
            json_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        
        # Append to output file
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json_line + '\n')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write to output file: {str(e)}")
            return
        
        # Move to next entry
        self.current_index += 1
        
        if self.current_index >= len(self.entries):
            QMessageBox.information(
                self,
                "Complete",
                f"All entries reviewed!\n\n"
                f"Accepted entries saved to:\n{self.output_file}"
            )
            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            self.text_edit.setPlainText("")
            self.status_label.setText("Review complete!")
        else:
            self.display_current_entry()
            
    def reject_entry(self):
        """Reject the current entry and move to next."""
        self.current_index += 1
        
        if self.current_index >= len(self.entries):
            QMessageBox.information(
                self,
                "Complete",
                f"All entries reviewed!\n\n"
                f"Accepted entries saved to:\n{self.output_file}"
            )
            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            self.text_edit.setPlainText("")
            self.status_label.setText("Review complete!")
        else:
            self.display_current_entry()


def main():
    app = QApplication(sys.argv)
    
    # Set dark theme for the application
    app.setStyleSheet("""
        QWidget {
            background-color: #2E2E2E;
            color: ivory;
        }
        QPushButton:hover {
            background-color: #4E4E4E;
        }
        QPushButton:pressed {
            background-color: #5E5E5E;
        }
    """)
    
    reviewer = JSONLReviewer()
    reviewer.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
