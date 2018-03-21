project('kastore', 'c')
add_global_arguments([
    '-std=c99', '-Wall', '-Werror', '-pedantic', '-W',
    '-Wno-unused-parameter', '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common'], language : 'c')
      
cunit_dep = dependency('cunit')
executable('tests', ['tests.c', 'kastore.c'], dependencies: cunit_dep)
