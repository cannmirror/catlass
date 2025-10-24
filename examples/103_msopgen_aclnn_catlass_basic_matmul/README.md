```bash
cd examples/103_msopgen
msopgen gen -i basic_matmul.json -c ai_core=Ascend910B1 -lan cpp -out CatlassBasicMatmul
cd CatlassBasicMatmul
./build.sh
./build_out/custom_opp*
```
