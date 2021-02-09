def buildUploadPypi():
    info = 'pip install build twine\n' \
           'pip install --upgrade build twine\n' \
           'python -m build\n' \
           'twine upload --skip-existing --repository pypi dist/*'

    print(info)
    return info
