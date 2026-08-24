# Next experiments

The current method (QPEFA + compressed Adam) saves persistent training memory and inference memory on a 95k-parameter decoder. Next work, in order:

1. **GPU scale.** Repeat the grammar (and then TinyStories) at 5.25M and 20M parameters. Confirm flip rates stay nonzero and that the val-loss gap versus FP32 does not reopen.
2. **Natural language.** WikiText-2 and TinyStories with a GPT-2 tokenizer. Report val PPL against FP32 Adam and a latent-weight STE (BitNet-pattern) baseline at matched step count.
3. **Packed GEMM.** Bit-serial 2-bit matmul so the ephemeral dequantized tile is not materialised. This is the remaining peak-memory term.
4. **W2A4.** Activation quantization below 8 bits, with the same QPEFA optimizer.
5. **Export.** Pack codes to 2 bits per weight on disk (the reference still stores uint8 values in `{0,1,2,3}` and reports packed-equivalent bytes).
