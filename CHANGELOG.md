# Changelog

## 0.1.0 (2026-10-09)


### ⚠ BREAKING CHANGES

* on_check callbacks now receive a CheckProgress object as their third argument.
* replace the moments import and legacy interfaces with momentous; measurements use raw Wigner-D moments and labeled covariance, and event samples take costheta and phi with explicit event grouping.

### Features

* First commit ([300c4bd](https://github.com/denehoffman/momentous/commit/300c4bddac3a551ddfb172460265d80018990a4f))
* Prepare publication and improve search results ([4085c4a](https://github.com/denehoffman/momentous/commit/4085c4a217306a47871e124cee20d2110649deab))
* Unify momentous analysis and acceptance-corrected extraction ([46e18cf](https://github.com/denehoffman/momentous/commit/46e18cf1d2445223fb7f5661a33df7cf130d2f96))


### Performance Improvements

* Speed up waveset searches and add check callbacks ([890e11c](https://github.com/denehoffman/momentous/commit/890e11cefa8fd603cbbe18d832d92b39609d5e6e))
