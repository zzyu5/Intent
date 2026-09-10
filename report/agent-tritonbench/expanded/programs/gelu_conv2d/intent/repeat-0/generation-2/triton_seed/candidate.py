import intent
import intent.language as I

def build(context):
    compiled = context.load_source('gelu_conv2d_nobias.py')

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate='none', out=None):
        if bias is not None:
            raise NotImplementedError('the fixed invocation profile has no bias')
        if stride != 1 or padding != 1 or dilation != 1 or (groups != 1) or (approximate != 'none'):
            raise NotImplementedError('unsupported parameters for the fixed invocation profile')
        if out is None:
            return compiled.run(input, weight)
        compiled(input, weight, out)
        return out
    return wrapper
