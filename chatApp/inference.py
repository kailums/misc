import time
from conversation import get_conv
from chatio import ChatIO, SimpleChatIO

def generate_stream_func(model, tokenizer, gen_params, context_len, echo=False, stream_interval=2):
    prompts = gen_params['prompts']
    max_new_tokens = gen_params['max_new_tokens']
    stop = gen_params['stop']
    stop_token_ids = gen_params['stop_token_ids']
    stop_token_ids.append(tokenizer.eos_token_id)
    device = model.device

    lhs_batch = tokenizer(prompts, padding=True)

    lhs_tokens = lhs_batch.to(device)
    lhs_masks = lhs_batch.to(device)

    past_kvs = None
    output_ids = list(lhs_batch.input_ids)
    input_echo_len = len(lhs_batch.input_ids)

    for i in range(max_new_tokens):
        with torch.no_grad():
            lhs_results = model.forward(
                lhs_tokens, lhs_masks, past_key_values=past_kvs)
        
        logits = lhs_results.logits
        past_kvs = lhs_results.past_key_values
        # greedy search
        lhs_tokens = torch.argmax(
            lhs_results.logits[:, -1, :], dim=1, keepdim=True)

        token = lhs_tokens[0].to('cpu')
        output_ids.append(token)

        if token in stop_token_ids:
            stoped = True
        else:
            stoped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stoped:
            if echo:
                tmp_output_ids = output_ids
            else:
                tmp_output_ids = output_ids[input_echo_len:]

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            yield {
              'text': output,
            }

        if stoped:
            break

        lhs_masks = torch.cat([lhs_masks, torch.ones((len(prompts), 1), device=device)], dim=1)

    yield {
      'text': output
    }

def chat_loop(
    model,
    tokenizer,
    max_new_tokens: int,
    chatio: ChatIO = None,
    debug: bool = False,
):
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

        gen_params = {
            "prompts": prompt,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(
            model,
            tokenizer,
            gen_params,
            context_len=context_len,
        )
        t = time.time()
        outputs = chatio.stream_output(output_stream)
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
