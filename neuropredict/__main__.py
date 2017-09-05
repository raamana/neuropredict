from sys import version_info

if version_info.major==2 and version_info.minor==7:
    import neuropredict
elif version_info.major > 2:
    from neuropredict import neuropredict
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')

def main():
    "Entry point."

    neuropredict.run()

if __name__ == '__main__':
    main()
