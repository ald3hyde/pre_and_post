# dump2glassanalysis.py
 - lammpsのMDシミュレーションで得られたdump (.lammpstrj) ファイルから
   配位数とX-O-X (X: ネットワークフォーマー) の計算を行うプログラム。
# dump2glassdescript.py
 - lammpsのMDシミュレーションで得られたdumpファイルから，配位数，$Q^n$
   ，ガラス構造を表現する構造記述子を計算するプログラムです。
 - スラブがある系で表面のみに着目したい場合は，最も露出しているSiから
   の距離 (CUTOFF (default: 5.0 Å))をプログラムで指定して，-s オプショ
   ンを追加する。
# glass_advanced_analysis.py
 - g(r)，角度分布，原始リング分布など，より発展的な構造解析を行うプログラムです。
 - `--cutoff Si-O=2.3 --cutoff Al-O=2.4` のようにカットオフを指定しながら，
   `--rdf-pair`, `--angle-triplet`, `--ring-max` オプションで解析の種類を選択できます。
 - `--processes` オプションを利用することで，複数のdumpファイルを並列(multiprocessing)
   に解析できます。
 - 解析結果はCSV形式で出力されるため，pandasや可視化ツールで簡単に扱えます。
# 詳細な情報
- https://amorphous.tf.chiba-u.jp/kayano/ まで。
