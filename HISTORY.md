# History

## 0.1.3

* Cap dependencies in order to avoid outside changes that caused `staganogan` to malfunctioned.

### Resolved Issues

* Issue #50: Cap pillow and other dependencies.
* Issue #55: Update reedsolo.

## 0.1.2

* Add option to work with a custom pretrained model from CLI and Python
* Refactorize Critics and Decoders to match Encoders code style
* Make old pretrained models compatible with new code versions
* Cleanup unneeded dependencies
* Extensive unit testing

## 0.1.1

* Add better pretrained models.
* Improve support for non CUDA devices.

## 0.1.0 - First release to PyPi

* SteganoGAN class which can be fitted, saved, loaded and used to encode and decode messages.
* Basic command line interface that allow using pretrained models.
* Basic and Dense pretrained models for demo purposes.
