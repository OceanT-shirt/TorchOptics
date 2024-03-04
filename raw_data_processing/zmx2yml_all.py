import os
import glob
from raw_data_processing import zmx2yml
import csv


def zex2yml_all():
    impdir = input("変換対象が入っているディレクトリを選択: ")
    expdir = input("出力を入れるディレクトリを選択: ")
    extension = ".zmx"  # 処理するファイルの拡張子を指定
    extension2 = ".ZMX"  # 処理するファイルの拡張子を指定

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(expdir, exist_ok=True)
    errors_list = []  # [e, file_name][]

    # ディレクトリ内のファイルをループ
    for file in glob.glob(os.path.join(impdir, '*')):
        if file.endswith(extension) or file.endswith(extension2):  # 拡張子がマッチする場合のみ処理を実行
            print("処理中:", file)
            filename = os.path.basename(file)  # パスからファイル名を取得
            output_file = os.path.join(expdir, f"{filename}.yml")
            try:
                zmx2yml.zmx2yml(file, output_file)
            except Exception as e:
                print("エラー:", e)
                print("処理をスキップします")
                errors_list.append([e, file])
                continue
    
    error_file = os.path.join(expdir, f"error.csv")
    with open(error_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for e in errors_list:
            writer.writerow(e)
