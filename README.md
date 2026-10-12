# MonoSLAM

A concise and thoroughly documented C++ implementation of the seminal work by [Davison et *al.* 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf), with parallel feature integration. The code has been recently reviewed and upgraded so that it is mostly compliant with ISO C++17; also, it compiles against OpenCV 5.

It runs at 30 Hz on commodity hardware; check the [sample videos](https://www.youtube.com/watch?v=zg3O0ZN1_rI&list=PLQjxkIPDI8hgJj4gUFwuYeaorkd8JWDpo).

Compile with:

```bash
cmake -S . -B build
cmake --build build
```
