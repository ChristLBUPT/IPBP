import gradio as gr
import os
import torch

def refresh_seeds(run_dir: str):
    # print(f'listing {run_dir}')
    seed_dirs = os.listdir(os.path.join('./models', run_dir)) 
    # seed_selector.choices = seed_dirs
    return gr.update(choices=seed_dirs, value=seed_dirs[0], interactive=True)
    # return {
    #     "__type__": "update",
    #     "choices": seed_dirs,
    # }

def get_sentences(run_name, seed_name):
    return [gr.Markdown(each) for each in open(os.path.join('models', run_name, seed_name, 'dev_result.txt'), 'r').read()]

# def dep_visualize_ui():
with gr.Blocks() as demo:

    with gr.Group(), gr.Row():
        file_selector = gr.Dropdown(os.listdir('./models'), label='select a folder of runs')
        seed_selector = gr.Dropdown(['None'], allow_custom_value=True, label='select a seed')
        file_selector.change(refresh_seeds, [file_selector], [seed_selector])
    
    with gr.Group(), gr.Row():
        with gr.Group(), gr.Row():
            prev_btn = gr.Button("👈", scale=1)
            page_input = gr.Textbox("1", label="Page", interactive=True, scale=3)
            go = gr.Button("☝", scale=1)

            next_btn = gr.Button("👉", scale=1)
        reload_btn = gr.Button("reload")
        with gr.Tab("sentences") as sentences_tab:
            dep_images = []
            for _ in range(5):
                dep_images.append(gr.Image("", language="markdown"))
                
            # print(file_selector.choices, seed_selector.value)
            # for each in open(os.path.join('models', file_selector.value, seed_selector.value, 'eval_results.txt'), 'r').read():
            #     gr.Markdown(each)
        
        reload_btn.click(get_sentences, [file_selector, seed_selector], [sentences_tab])
            
            

    demo.launch(server_name="0.0.0.0", server_port=8416,)

# if __name__ == "__main__":
#     dep_visualize_ui()
        