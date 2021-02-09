def className(o):
    import inspect
    return o.__name__ if inspect.isclass(o) else o.__class__.__name__


def tupleAppend(t, o):
    return (*t, o)
