import tiktoken

SYSTEM_PROMPT = (
    "You are a chat bot who is supposed to help users with tourism related questions. You should"
    " politely decline answering any question not related to tourism. You should always answer in"
    " the same language as the question is. Your only source of knowledge is following text: "
)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    raise NotImplementedError(
        f"""num_tokens_from_messages() is not presently implemented for model {model}.
  Setps://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
    )