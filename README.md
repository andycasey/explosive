Fireworks
==========
[![Build Status](https://travis-ci.org/andycasey/fireworks.svg?branch=master)](https://travis-ci.org/andycasey/fireworks)

Detailed chemical abundances using (an extension of) [The Cannon](http://adsabs.harvard.edu/abs/2015ApJ...808...16N).

Contributions
-------------

First clone the repository:

    git clone git@github.com:andycasey/fireworks.git

The article includes references to the `git` hash, so use the included post-update hooks to keep
these fresh:

    cd fireworks
    cat hooks/post-commit | tee .git/hooks/post-{commit,update,merge} >/dev/null
    chmod +x .git/hooks/post-{commit,update,merge}

The article can be compiled with the following commands:

    cd article
    make


Authors
-------
- *Andrew R. Casey (Cambridge)*
- *Melissa K. Ness (MPIA)*
- *David W. Hogg (NYU/MPIA)*
- *Gerry Gilmore (Cambridge)*
- *Hans-Walter Rix (MPIA)*

Copyright (2015) the authors. All rights reserved.
