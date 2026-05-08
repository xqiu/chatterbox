import gradio as gr
import os
import sys
import importlib.util
from i18n import install, available_languages
from utils import get_audio_files

# Import process_audio_files from dir_vc.py dynamically
spec = importlib.util.spec_from_file_location("dir_vc", os.path.join(os.path.dirname(__file__), "dir_vc.py"))
dir_vc = importlib.util.module_from_spec(spec)
sys.modules["dir_vc"] = dir_vc
spec.loader.exec_module(dir_vc)

# Install default language at startup
install("en")
_ = _

# Streaming log output to Gradio UI
import threading
import queue
import time
import contextlib
import sys
import webbrowser

class StreamCatcher:
    def __init__(self):
        self.q = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
    def write(self, data):
        self.q.put(data)
    def flush(self):
        pass
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

def run_voice_conversion(input_dir, output_dir, target_voice_path, progress=gr.Progress(track_tqdm=True)):
    def generator():
        catcher = StreamCatcher()
        logs = ""
        output_files = []
        try:
            with catcher:
                t = threading.Thread(target=lambda: output_files.extend(
                    dir_vc.process_audio_files(input_dir, output_dir, target_voice_path, continue_job=False)
                ))
                t.start()
                while t.is_alive() or not catcher.q.empty():
                    while not catcher.q.empty():
                        chunk = catcher.q.get()
                        logs += chunk
                        # 返回顺序要和 outputs 对齐（见第 3 步），downloads 用 gr.update() 占位
                        yield (
                            True,
                            gr.update(value=f"<pre>{logs}</pre>"),  # result (HTML)
                            gr.update(value=None),                  # downloads (Files) —— 只占位
                            gr.update(interactive=False)            # run_btn
                        )
                    time.sleep(1)
                t.join()
            
            # ⬇️ 处理完成：同时更新 HTML 和 downloads
            if not output_files:
                msg = f"No output files generated.\n<pre>{logs}</pre>"
                yield False, gr.update(value=msg), gr.update(value=[]), gr.update(interactive=True)
            else:
                abs_paths = [os.path.abspath(f) for f in output_files]
                links_html = "<br>".join(f"<div>{p}</div>" for p in abs_paths)
                result_html = f"<h2>Processing complete.</h2> Output files:<br>{links_html}<br><pre>{logs}</pre>"

                # ✅ downloads 传入“文件路径列表”
                # Handle downloads: only allow files in cwd or temp, else leave empty
                if len(abs_paths) == 1 and os.path.dirname(abs_paths[0]) in (os.getcwd(), os.getenv('TEMP'), os.getenv('TMP')):
                    downloads_update = gr.update(value=abs_paths[0])
                elif all(os.path.dirname(p) in (os.getcwd(), os.getenv('TEMP'), os.getenv('TMP')) for p in abs_paths):
                    downloads_update = gr.update(value=abs_paths)
                else:
                    downloads_update = gr.update(value=[])
                yield (
                    True,
                    gr.update(value=result_html),        # result
                    downloads_update,                    # downloads
                    gr.update(interactive=True)           # run_btn
                )
        except Exception as e:
            err = f"Error: {str(e)}\n<pre>{logs}</pre>"
            # 失败时清空 downloads
            yield False, gr.update(value=err), gr.update(value=[]), gr.update(interactive=True)
    return generator

def update_target_files(target_dir):
    files = get_audio_files(target_dir)
    return gr.update(choices=files, value=files[0] if files else None)

# BEGIN File dialog functions
def open_system_explorer(path_to_open):
    # Open the path in the default system file browser
    webbrowser.open(f"file://{path_to_open}")
    return f"Opening {path_to_open} in system explorer..."

def open_target_folder_in_system_explorer():
    homefolder = os.environ.get('userprofile')
    msg = open_system_explorer(os.path.join(homefolder, "chatterbox/test/target"))
    print(msg)

def open_select_dir_dialog(preferred_dir: str = None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(initialdir=preferred_dir)

def open_select_file_dialog(extension: str = None, preferred_dir: str = None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(initialdir=preferred_dir, filetypes=[(f"{extension} files", f"*.{extension}")])
#END file dialog functions
# BEGIN shown success/fail
def on_infer_button_click():
    """right after button clicked, change ui status

    Returns:
        status of changed ui elements
    """
    return gr.update(visible="hidden"), gr.update(visible="hidden")

def handle_infer_ui_result(success):
    """
    Handle the result of the inference UI operation.
    
    :param success: Boolean indicating if the operation was successful.
    :param result: The output message from the operation.
    :return: Tuple of updates for success and error spans, and the result text area.
    """
    #check type of success
    print(f'try update success as {success}, {not success}')
    v_value = True if success else "hidden"
    return gr.update(visible=v_value), gr.update(visible=not v_value)
# END shown success/fail

def gradio_ui(target_dir):
    def on_change_language(lang):
        trans = install(lang)
        _ = trans.gettext
        return (
            gr.update(value=_('# Chatterbox Voice Conversion Batch Tool')),
            gr.update(label=_('Input File or Directory'), placeholder=_('Path to input audio file or directory with audio files')),
            gr.update(label=_('Output Directory'), placeholder=_('Path to save output files')),
            gr.update(value=_('Refresh Target Files List')),
            gr.update(value=_('Open Target Files Folder')),
            gr.update(label=_('Target Voice File (from target dir)')),
            gr.update(value=_('Run Voice Conversion')),
            gr.update(label=_('Result')),
            gr.update(label=_('Language'))
        )

    def on_refresh_files(_click=None):
        return gr.update(choices=get_audio_files(target_dir))

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                title_md = gr.Markdown(_('# Chatterbox Voice Conversion Batch Tool'))
            with gr.Column(scale=1):
                lang_dd = gr.Dropdown(
                    choices=available_languages(),
                    value="en",
                    label=_('Language'),
                    scale=0
                )
        with gr.Row():
            input_dir = gr.Textbox(
                label=_('Input File or Directory'),
                placeholder=_('Path to input audio file or directory with audio files')
            )
            input_browse_button = gr.Button(_('Browse'))
            input_browse_button.click(
                fn=open_select_dir_dialog,
                inputs=input_dir,
                outputs=input_dir
            )
        with gr.Row():
            output_dir = gr.Textbox(
                label=_('Output Directory'),
                placeholder=_('Path to save output files')
            )
            output_browse_button = gr.Button(_('Browse'))
            output_browse_button.click(
                fn=open_select_dir_dialog,
                inputs=output_dir,
                outputs=output_dir
            )
        with gr.Row():
            refresh_btn = gr.Button(_('Refresh Target Files List'))
            open_target_btn = gr.Button(_('Open Target Files Folder'))
        target_file = gr.Dropdown(
            label=_('Target Voice File (from target dir)'),
            choices=get_audio_files(target_dir),
            interactive=True
        )
        run_btn = gr.Button(_('Run Voice Conversion'))
        with gr.Row():
            with gr.Column(scale=1):
                success_span = gr.HTML(label=_("Success"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 green;'>✔️</span>", visible="hidden")
                error_span = gr.HTML(label=_("Error"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 red;'>❌</span>", visible="hidden")
                # full_log_path = gr.Textbox(label=_("Log"), value=log_file, interactive=False).style(show_copy_button=True)
            is_success = gr.Checkbox(visible=False)
            with gr.Column(scale=4):
                result = gr.HTML(label=_('Result'), elem_id="result_html")
        try:
            downloads = gr.Files(label="Download Outputs")
        except Exception:
            downloads = gr.File(label="Download Outputs", file_count="multiple")
        gr.HTML("""
        <style>
        #result_html {
            min-height: 200px;
        }
        /* Hide Gradio footer and settings */
        footer, .svelte-1ipelgc, .svelte-1ipelgc *, .gradio-container .fixed.bottom-4.right-4, .gradio-container .fixed.bottom-4.left-4 {
            display: none !important;
        }
        </style>
        """)

        lang_dd.change(
            on_change_language,
            inputs=lang_dd,
            outputs=[title_md, input_dir, output_dir, refresh_btn, open_target_btn, target_file, run_btn, result, lang_dd]
        )
        refresh_btn.click(
            on_refresh_files,
            inputs=None,
            outputs=target_file
        )
        open_target_btn.click(
			open_target_folder_in_system_explorer,
			inputs=None,
			outputs=[]
        )

        # 1) On load: restore values from localStorage into the components (including language)
        demo.load(
            fn=None,
            inputs=None,
            outputs=[lang_dd, input_dir, output_dir, target_file],
            js="""
        () => {
            const lang = localStorage.getItem('vc_lang') || "en";
            const inDir = localStorage.getItem('vc_input_dir') || "";
            const outDir = localStorage.getItem('vc_output_dir') || "";
            const target = localStorage.getItem('vc_target_file') || null;
            return [lang, inDir, outDir, target];
        }
        """
        )
        lang_dd.change(
            fn=None, inputs=lang_dd, outputs=None,
            js="(v) => { if (v !== undefined && v !== null) localStorage.setItem('vc_lang', v); }"
        )
        input_dir.change(
            fn=None, inputs=input_dir, outputs=None,
            js="(v) => { localStorage.setItem('vc_input_dir', v ?? ''); }"
        )
        output_dir.change(
            fn=None, inputs=output_dir, outputs=None,
            js="(v) => { localStorage.setItem('vc_output_dir', v ?? ''); }"
        )
        target_file.change(
            fn=None, inputs=target_file, outputs=None,
            js="(v) => { if (v !== undefined && v !== null) localStorage.setItem('vc_target_file', v); }"
        )

        def on_run(input_dir, output_dir, target_file):
            if not input_dir or not output_dir or not target_file:
                yield gr.update(value=_('Please provide all required fields.')), gr.update(value=[]), gr.update(interactive=True)
                return
            target_voice_path = os.path.join(target_dir, target_file)
            gen = run_voice_conversion(input_dir, output_dir, target_voice_path)
            for update in gen():
                yield update

        run_btn.click(
            on_infer_button_click,
            inputs=None,
            outputs=[success_span, error_span],
        ).then(
            on_run,
            inputs=[input_dir, output_dir, target_file],
            outputs=[is_success, result, downloads, run_btn],
            preprocess=False,
            show_progress=True,
            queue=True,
        ).then(
            fn=handle_infer_ui_result,
            inputs=[is_success],
            outputs=[success_span, error_span]
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chatterbox Gradio VC Batch UI")
    parser.add_argument('--target_dir', type=str, required=True, help='Directory containing target wav/mp3 files')
    args = parser.parse_args()
    demo = gradio_ui(args.target_dir)
    demo.launch(share=False, inbrowser=True)
    
