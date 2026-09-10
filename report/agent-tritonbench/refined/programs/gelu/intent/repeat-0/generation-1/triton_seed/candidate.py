import intent
import intent.language as I

def build(context):
    exact = context.load_source('gelu_none.py')
    tanh = context.load_source('gelu_tanh.py')

    def wrapper(input, approximate='none'):
        if approximate == 'none':
            return exact.run(input)
        if approximate == 'tanh':
            return tanh.run(input)
        raise ValueError("approximate must be 'none' or 'tanh'")
    return wrapper
