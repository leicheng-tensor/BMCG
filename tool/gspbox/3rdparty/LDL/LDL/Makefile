#------------------------------------------------------------------------------
# LDL Makefile
#------------------------------------------------------------------------------

VERSION = 2.1.0

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

demos: all

# Compile all C code, including the C-callable routine.
all:
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )

# compile just the C-callable libraries (not Demos)
library:
	( cd Lib    ; $(MAKE) )

# remove object files, but keep the compiled programs and library archives
clean:
	( cd Lib    ; $(MAKE) clean )
	( cd Demo   ; $(MAKE) clean )
	( cd Doc    ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )

# clean, and then remove compiled programs and library archives
purge:
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd Doc    ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) *.mex* )

distclean: purge

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

ccode: library

lib: library

# install LDL
install:
	$(CP) Lib/libldl.a $(INSTALL_LIB)/libldl.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libldl.$(VERSION).a libldl.a )
	$(CP) Include/ldl.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libldl*.a
	chmod 644 $(INSTALL_INCLUDE)/ldl.h

# uninstall LDL
uninstall:
	$(RM) $(INSTALL_LIB)/libldl*.a
	$(RM) $(INSTALL_INCLUDE)/ldl.h

