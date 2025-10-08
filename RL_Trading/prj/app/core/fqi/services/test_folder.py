import errno, os, stat, shutil
import matplotlib.pyplot as plt

def handleRemove(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

if __name__ == '__main__':

    #path = "C:/Users/Riccardo/Desktop/seed15723"
    path = "C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/results/test3/tmp/seed15823"
    #os.umask(0)
    #os.makedirs(path, exist_ok=True)
    series = [0,1,2,3,4,5,6,7,8,9]
    plt.figure()
    plt.plot(series)
    #plt.savefig(path+'/actions.png')
    plt.savefig(os.path.join(path, f'Actions_iter.png'))
    #plt.show()
    shutil.rmtree(path, onerror=handleRemove)
