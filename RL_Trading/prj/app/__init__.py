import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
    __package_name__ = pkg_resources.get_distribution(__name__).project_name
except:
    __version__ = 'unknown'
    __package_name__ = 'unknown'
