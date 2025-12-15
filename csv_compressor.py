import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import random
import os
import threading
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='csv_compressor.log',
    encoding='utf-8'
)

class CSVCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("超大CSV流式采样压缩工具 (Reservoir Sampling)")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # 样式配置
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", font=("Microsoft YaHei", 10))
        self.style.configure("TButton", font=("Microsoft YaHei", 10))
        
        # 变量
        self.file_path = tk.StringVar()
        self.target_rows = tk.IntVar(value=5000)
        self.status_msg = tk.StringVar(value="准备就绪")
        self.is_processing = False
        self.progress_var = tk.DoubleVar(value=0)
        
        self.create_widgets()

    def create_widgets(self):
        # 容器
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区
        title_label = tk.Label(main_frame, text="CSV 数据特征保留压缩", font=("Microsoft YaHei", 16, "bold"), fg="#333")
        title_label.pack(pady=(0, 20))

        # 文件选择区
        file_frame = ttk.LabelFrame(main_frame, text="1. 选择源文件 (支持GB级大文件)", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        entry_file = ttk.Entry(file_frame, textvariable=self.file_path, state="readonly")
        entry_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        btn_browse = ttk.Button(file_frame, text="浏览...", command=self.browse_file)
        btn_browse.pack(side=tk.RIGHT)

        # 参数设置区
        setting_frame = ttk.LabelFrame(main_frame, text="2. 设置采样参数", padding="10")
        setting_frame.pack(fill=tk.X, pady=10)
        
        lbl_rows = ttk.Label(setting_frame, text="目标保留行数:")
        lbl_rows.pack(side=tk.LEFT)
        
        spin_rows = ttk.Spinbox(setting_frame, from_=100, to=1000000, textvariable=self.target_rows, width=15)
        spin_rows.pack(side=tk.LEFT, padx=10)
        
        lbl_hint = ttk.Label(setting_frame, text="(算法将自动随机抽取，保留整体分布)", foreground="gray", font=("Microsoft YaHei", 8))
        lbl_hint.pack(side=tk.LEFT)

        # 进度条区
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=20)
        
        self.lbl_status = ttk.Label(progress_frame, textvariable=self.status_msg, anchor="center")
        self.lbl_status.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, ipady=5)

        # 按钮区
        self.btn_start = ttk.Button(main_frame, text="开始压缩处理", command=self.start_processing_thread)
        self.btn_start.pack(fill=tk.X, ipady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            self.file_path.set(filename)

    def start_processing_thread(self):
        """启动后台线程，避免界面卡死"""
        input_file = self.file_path.get()
        k_rows = self.target_rows.get()

        if not input_file:
            messagebox.showwarning("提示", "请先选择一个CSV文件")
            return
        
        if k_rows <= 0:
            messagebox.showwarning("提示", "目标行数必须大于0")
            return

        self.is_processing = True
        self.btn_start.config(state="disabled")
        
        # 使用 threading 将计算任务移出主线程
        t = threading.Thread(target=self.process_algorithm, args=(input_file, k_rows))
        t.daemon = True # 设为守护线程，关闭窗口时自动结束
        t.start()

    def process_algorithm(self, input_path, k):
        """
        蓄水池采样算法核心逻辑 (Reservoir Sampling)
        """
        try:
            logging.info(f"开始处理文件: {input_path}, 目标行数: {k}")
            
            # 检查文件是否存在
            if not os.path.exists(input_path):
                logging.error(f"文件不存在: {input_path}")
                self.update_ui_finish(False, "文件不存在")
                return
                
            file_size = os.path.getsize(input_path)
            logging.info(f"文件大小: {file_size:,} 字节")
            
            reservoir = []
            header = []
            
            # 生成输出文件名
            dir_name, base_name = os.path.split(input_path)
            name_part, ext_part = os.path.splitext(base_name)
            timestamp = int(time.time())
            output_path = os.path.join(dir_name, f"{name_part}_sampled_{timestamp}{ext_part}")
            logging.info(f"输出文件将保存至: {output_path}")

            # 优化读取：使用 utf-8-sig 以兼容 Excel (处理 BOM)
            # newline='' 是 csv 模块的最佳实践
            with open(input_path, 'r', encoding='utf-8-sig', newline='', errors='replace') as f_in:
                reader = csv.reader(f_in)
                
                # 读取表头
                try:
                    header = next(reader)
                    logging.info(f"成功读取表头，包含 {len(header)} 个字段")
                except StopIteration:
                    logging.warning(f"文件为空: {input_path}")
                    self.update_ui_finish(False, "空文件")
                    return

                # 初始化计数器
                row_index = 0
                
                # --- 流式处理循环 ---
                # 这是一个极其节省内存的循环，无论文件多大，
                # 内存中只保留 'k' 行数据 + 当前读取的一行缓冲
                for row in reader:
                    # 进度条更新逻辑 (使用行计数，每2000行更新一次)
                    if row_index % 2000 == 0:
                        # 由于无法可靠获取文件位置，使用行计数作为进度参考
                        # 对于未知总行数的文件，进度条会平稳增长
                        progress = min(95.0, (row_index / (k * 10)) * 100) if k > 0 else 0
                        self.update_ui_progress(progress, row_index)

                    # 蓄水池算法核心 (Algorithm R)
                    if row_index < k:
                        # 阶段1: 填满蓄水池
                        reservoir.append(row)
                    else:
                        # 阶段2: 随着 row_index 增加，新元素进入池子的概率逐渐降低 (k / row_index)
                        # 这保证了所有元素被选中的概率最终都为 k / N
                        
                        # 生成一个 0 到 row_index 的随机整数
                        j = random.randint(0, row_index)
                        if j < k:
                            reservoir[j] = row
                    
                    row_index += 1

            logging.info(f"完成数据读取，总处理行数: {row_index}")
            
            # 处理只有表头的情况
            if row_index == 0:
                logging.warning(f"文件只有表头，没有数据行: {input_path}")
                # 仍然保存文件，但只有表头
                with open(output_path, 'w', encoding='utf-8-sig', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header)
                logging.info(f"已保存只有表头的文件: {output_path}")
                self.update_ui_finish(True, f"成功！\n文件只有表头，已保存至: {output_path}")
                return
            
            # 处理实际行数少于目标行数的情况
            if row_index < k:
                logging.info(f"实际行数 ({row_index}) 少于目标行数 ({k})，将保存所有行")
            
            # 写入文件
            self.root.after(0, lambda: self.status_msg.set("正在写入新文件..."))
            try:
                with open(output_path, 'w', encoding='utf-8-sig', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header)
                    writer.writerows(reservoir)
                    logging.info(f"成功写入输出文件: {output_path}, 写入行数: {len(reservoir)}")
            except PermissionError:
                logging.error(f"没有写入权限: {output_path}")
                self.update_ui_finish(False, "没有写入权限，请检查目标文件夹权限")
                return
            except Exception as write_e:
                logging.error(f"写入文件失败: {type(write_e).__name__}: {str(write_e)}")
                self.update_ui_finish(False, f"写入文件失败\n\n{type(write_e).__name__}: {str(write_e)}")
                return

            self.update_ui_finish(True, f"成功！\n原始行数估计: {row_index}\n已保存至: {output_path}")

        except Exception as e:
            error_msg = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}\n\n详细堆栈: {traceback.format_exc()}"
            logging.error(f"处理失败: {error_msg}")
            self.update_ui_finish(False, f"处理失败\n\n{type(e).__name__}: {str(e)}")

    def update_ui_progress(self, percent, rows_count):
        """线程安全地更新UI"""
        # 使用 root.after 或者 invoke 确保在主线程执行
        self.root.after(0, lambda: self._update_progress_impl(percent, rows_count))

    def _update_progress_impl(self, percent, rows_count):
        self.progress_var.set(percent)
        self.status_msg.set(f"正在扫描并采样... 已处理 {rows_count} 行 ({percent:.1f}%)")

    def update_ui_finish(self, success, msg):
        """处理结束后的回调"""
        self.root.after(0, lambda: self._finish_impl(success, msg))

    def _finish_impl(self, success, msg):
        self.is_processing = False
        self.btn_start.config(state="normal")
        self.progress_var.set(100 if success else 0)
        self.status_msg.set("处理完成" if success else "出错")
        
        if success:
            messagebox.showinfo("完成", msg)
        else:
            messagebox.showerror("错误", f"处理过程中出错: {msg}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVCompressorApp(root)
    root.mainloop()