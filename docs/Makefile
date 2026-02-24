BUILDDIR      = _build
DOXYGEN_XML   = doxygen/xml

all: ${DOXYGEN_XML}
	uv --project=../python run jupyter-book build -Wn --keep-going .

${DOXYGEN_XML}: ../c/kastore.h
	cd doxygen && doxygen

clean:
	rm -fR $(BUILDDIR) $(DOXYGEN_XML)
