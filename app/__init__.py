# Self-healing runtime patch for LangChain package compatibility in custom environments
try:
    import langchain
    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
except ImportError:
    pass
