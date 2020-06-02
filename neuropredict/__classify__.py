from sys import version_info

if version_info.major > 2:
    from neuropredict import classify
else:
    raise NotImplementedError('neuropredict requires Python 3 or higher. '
                              'Upgrade to Python 3+ or use [virtual] environments.')

def main():
    "Entry point."

    classify.cli()

if __name__ == '__main__':
    main()
