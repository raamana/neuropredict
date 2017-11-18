from sys import version_info

if version_info.major > 2:
    from neuropredict import run_workflow
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')

def main():
    "Entry point."

    run_workflow.cli()

if __name__ == '__main__':
    main()
