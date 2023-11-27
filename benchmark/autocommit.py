import subprocess


def autocommit(working_branch="benchmark"):
    # check that source branch is actually the current branch
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )
    if branch != working_branch:
        raise ValueError(
            "working_branch is different from the branch of the working directory"
        )

    # saves the current state of the directory into branch
    stash_msg = subprocess.check_output(["git", "stash"]).decode("ascii").strip()

    if stash_msg == "No local changes to save":
        # saves commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    else:
        # switch to detached head
        subprocess.check_call(["git", "switch", "--detach"])

        # updates with commited (merge) and uncommited (stash) modifications
        subprocess.check_call(["git", "stash", "apply"])

        # commits changes
        try:
            subprocess.check_output(["git", "commit", "-am", "experiment"])
        except:
            # I am assuming that the "commit" command failed because there was nothing
            # new to commit
            pass

        # saves commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )

        # switches back to original branch
        subprocess.check_call(["git", "switch", working_branch])

        # reverts working directory (including uncommited changes)
        subprocess.check_call(["git", "stash", "pop"])

    return commit_hash