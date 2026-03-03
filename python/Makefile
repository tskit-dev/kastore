
all: ext3

ext3: _kastoremodule.c
	CFLAGS="-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	       uv run python setup.py build_ext --inplace

ctags:
	ctags lib/*.c lib/*.h kastore/*.py

clean:
	rm -f *.so *.o tags
	rm -fR build
