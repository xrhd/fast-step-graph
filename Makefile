.PHONY: test test/show

test:
	uv run pytest

test/show:
	PLOT_MATRICES=true uv run pytest 