import time
import torch
from .conversation import get_conv
from .chatio import ChatIO, SimpleChatIO


def chat_loop(
    model,
    tokenizer,
    max_new_tokens: int = 512,
    generate_stream_func = None,
    generate_func = None,
    chatio: ChatIO = None,
    debug: bool = False,
):
    assert generate_stream_func is not None or generate_func is not None, 'should set generate function.'

    # Set context length
    context_len = model.config.max_position_embeddings

    # Chat
    def new_chat():
        conv = get_conv('llama2')
        return conv

    if chatio is None:
        chatio = SimpleChatIO()

    conv = None

    while True:
        if not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break

        if inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        chatio.prompt_for_output(conv.roles[1])
        if generate_stream_func is not None:
            output_stream = generate_stream_func(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                context_len=context_len,
            )
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())
        else:
            outputs = generate_func(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                context_len=context_len
            )
            outputs = chatio.output(outputs)
            t = time.time()
            duration = time.time() - t
            conv.update_last_message(outputs.strip())

        if debug:
            num_tokens = len(tokenizer.encode(outputs))
            msg = {
                "conv_template": conv.name,
                "prompt": prompt,
                "outputs": outputs,
                "speed (token/s)": round(num_tokens / duration, 2),
            }
            print(f"\n{msg}\n")
