pkgcheck <- function( pkg ) {
  
  # check if packages are not intalled and install them
  if (sum(!pkg %in% installed.packages()[, 'Package'])) {
    install.packages(
      pkg[which(!pkg %in% installed.packages()[, 'Package']) ],
                      dependencies = T)
  }
  
  # load all packages
  sapply(pkg, require, character.only = T)
  
}