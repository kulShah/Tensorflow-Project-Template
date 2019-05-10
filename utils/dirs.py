import os

def create_dirs(dirs):
    """
    param: a list of directories to create if these directories are not found
    return: exit code: 0:success, -1:failed
    """

    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        
        return 0
    except Exception as err:
        print( "Creating directories error: {0}".format(err) )
        exit(-1)