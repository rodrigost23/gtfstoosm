import nox

# Tell Nox to use uv for all sessions
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.12", "3.13"])
def tests(session):
    """Run tests with pytest across multiple Python versions."""
    # Install the package with test dependencies
    session.install(".")
    session.install("pytest")
    session.run("python", "-m", "pytest")


@nox.session(python="3.13")
def lint(session):
    """Run linting checks."""
    session.install(".")
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session(python="3.13")
def type_check(session):
    """Run type checking with mypy."""
    session.install(".")
    session.install("mypy")
    session.run("mypy", ".")
