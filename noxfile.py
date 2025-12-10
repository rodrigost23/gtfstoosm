import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session):
    """Run tests with pytest across multiple Python versions."""
    session.install("pytest")
    session.install("-r", "requirements.txt")
    session.run("python", "-m", "pytest")
