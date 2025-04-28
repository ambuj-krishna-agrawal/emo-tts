import gradio as gr
import logging, os

logging.basicConfig(level=logging.INFO)

def echo(text):
    logging.info("Echo called with: %s", text)
    return f"✅ You typed: {text}"

demo = gr.Interface(
    fn=echo,
    inputs=gr.Textbox(label="Say something"),
    outputs=gr.Textbox(),
    title="Tiny test"
)
demo.launch(
    share=True,           # open a public *.gradio.live URL
    server_name="0.0.0.0",
    server_port=7861,
    debug=True,
    inbrowser=False
)