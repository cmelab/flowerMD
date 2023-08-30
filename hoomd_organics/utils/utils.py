def check_return_iterable(obj):
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, str):
        return [obj]
    try:
        iter(obj)
        return obj
    except:  # noqa: E722
        return [obj]
