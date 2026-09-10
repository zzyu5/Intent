import intent
import intent.language as I

def build(context):
    partial_artifact = context.load_source('add_mean_partials.py')
    finish_artifact = context.load_source('add_mean_finish.py')

    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        partials = partial_artifact.run(input, other)
        return finish_artifact.run(partials).view(())
    return wrapper
