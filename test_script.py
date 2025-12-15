#!/usr/bin/env python3
"""
Test script for CSVCompressorApp core functionality
"""
import sys
import os
import time

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the core algorithm functionality
from csv_compressor import CSVCompressorApp
import tkinter as tk

class TestApp:
    """Test class to simulate the GUI app for testing"""
    def __init__(self):
        # Create a minimal root window for Tkinter compatibility
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window
        
        # Create the app instance
        self.app = CSVCompressorApp(self.root)
        
        # Override UI update methods to print instead
        self.original_update_ui_finish = self.app.update_ui_finish
        self.app.update_ui_finish = self.test_update_ui_finish
        
        self.success = False
        self.result_message = ""
    
    def test_update_ui_finish(self, success, msg):
        """Override to capture results"""
        self.success = success
        self.result_message = msg
        print(f"测试结果: {'成功' if success else '失败'}")
        print(f"消息: {msg}")
        # Call original method
        self.original_update_ui_finish(success, msg)
    
    def test_compression(self, input_file, target_rows):
        """Test the compression functionality"""
        print(f"\n=== 测试开始 ===")
        print(f"输入文件: {input_file}")
        print(f"目标行数: {target_rows}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Call the core algorithm directly
            self.app.process_algorithm(input_file, target_rows)
            
            # Wait a bit for processing to complete
            time.sleep(1)
            
            # Check if any output files were created
            dir_name = os.path.dirname(input_file)
            base_name = os.path.basename(input_file)
            name_part, ext_part = os.path.splitext(base_name)
            
            # List all files in the directory to find the output file
            output_files = []
            for file in os.listdir(dir_name):
                if file.startswith(f"{name_part}_sampled_") and file.endswith(ext_part):
                    output_files.append(os.path.join(dir_name, file))
            
            # Sort by modification time (newest first)
            output_files.sort(key=os.path.getmtime, reverse=True)
            
            if output_files:
                latest_output = output_files[0]
                print(f"\n输出文件已创建: {latest_output}")
                
                # Count lines in output file
                with open(latest_output, 'r', encoding='utf-8-sig') as f:
                    output_lines = sum(1 for _ in f)
                print(f"输出文件行数: {output_lines}")
                
                # Verify the output has the correct structure
                with open(latest_output, 'r', encoding='utf-8-sig') as f:
                    import csv
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        print(f"输出文件包含表头: {header}")
                    except StopIteration:
                        print("警告: 输出文件没有表头")
            
            return self.success
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            print(f"\n=== 测试结束 ===")
            # Cleanup
            self.root.destroy()

if __name__ == "__main__":
    # Test with our sample file
    test_file = "test_data.csv"
    
    if os.path.exists(test_file):
        test_app = TestApp()
        
        # Test with target rows greater than actual data
        print("\n=== 测试1: 目标行数大于实际数据 ===")
        test_app.test_compression(test_file, 20)
        
        # Test with target rows less than actual data
        print("\n=== 测试2: 目标行数小于实际数据 ===")
        test_app.test_compression(test_file, 5)
        
        # Test with target rows equal to 1 (edge case)
        print("\n=== 测试3: 目标行数为1 ===")
        test_app.test_compression(test_file, 1)
        
    else:
        print(f"测试文件不存在: {test_file}")
