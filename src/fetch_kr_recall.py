"""소비자24 리콜정보 OpenAPI - 자동차(메뉴ID 0301) 전체 수집 -> CSV

사용법
    # 키는 코드에 쓰지 말고 환경변수로 넘기기 (GitHub에 올라가지 않게)
    Windows PowerShell:  $env:CONSUMER24_KEY="발급받은키"; python src/fetch_kr_recall.py
    Mac/Linux:           CONSUMER24_KEY=발급받은키 python src/fetch_kr_recall.py

    옵션: --start 20100101 --end 20261231 (리콜 공표일 범위), --out data/kr_recall_auto.csv
출처: 소비자24 OpenAPI 개발가이드(리콜정보) V2.5
"""
import argparse, csv, os, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

URL = "https://www.consumer.go.kr/openapi/recall/contents/index.do"
FIELDS = ["recallSn", "cntntsId", "productNm", "makr", "bsnmNm", "mnfcturPd", "modlNmInfo", "mnfcturNoInfo",
          "stdBrcd", "distbTmlmtDe", "prmisnNo", "mdlpClNo", "aditfield13", "etcInfo", "mainSleoffic", "shrtcomCn",
          "recallSe", "recallPublictBgnde", "recallPublictEndde", "injryCauseResult", "injryFrgltyTrgter", "hrmflGrad",
          "acdntCn", "cnsmrGhvrTips", "trtmntAtpn", "recallBgnde", "recallEndde", "recallProcssInfo", "recallEntrpsInfo",
          "infoOriginInstt", "infoOriginUrl", "infoCreatInstt", "infoCreatUrl", "recallImgUrls"]
ERR = {"10": "어플리케이션 에러", "20": "DB 에러", "30": "데이터 없음", "40": "잘못된 요청 파라미터",
       "41": "페이지당 요청 개수 초과", "50": "등록되지 않은 서비스키", "51": "등록되지 않은 사용자", "99": "기타 에러"}


def local(tag):  # 네임스페이스 제거
    return tag.split("}", 1)[-1]


def fetch_page(key, page, per, start=None, end=None, retries=3):
    q = {"serviceKey": key, "pageNo": page, "cntPerPage": per, "cntntsId": "0301"}
    if start: q["recallPublictBgnde"] = start
    if end: q["recallPublictEndde"] = end
    url = URL + "?" + urllib.parse.urlencode(q)
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            if i == retries - 1: raise
            print(f"  page {page} 재시도 ({e})"); time.sleep(2 * (i + 1))


def parse(root):
    code, msg, total, rows = None, None, 0, []
    for el in root.iter():
        t = local(el.tag)
        if t == "code": code = (el.text or "").strip()
        elif t == "codeMsg": msg = (el.text or "").strip()
        elif t == "allCnt": total = int((el.text or "0").strip() or 0)
        elif t == "content":
            row = {f: "" for f in FIELDS}
            for ch in el:
                n = local(ch.tag)
                if n in row: row[n] = (ch.text or "").strip()
            rows.append(row)
    return code, msg, total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("CONSUMER24_KEY"))
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--out", default="data/kr_recall_auto.csv")
    a = ap.parse_args()
    if not a.key: sys.exit("서비스키가 없어요. CONSUMER24_KEY 환경변수나 --key로 넘겨주세요.")

    code, msg, total, rows = parse(fetch_page(a.key, 1, a.per, a.start, a.end))
    if code not in (None, "00"): sys.exit(f"오류 {code}: {ERR.get(code, msg)}")
    pages = -(-total // a.per)
    print(f"전체 {total}건, {pages}페이지")
    seen = {r["recallSn"] for r in rows}
    for p in range(2, pages + 1):
        c, m, _, rs = parse(fetch_page(a.key, p, a.per, a.start, a.end))
        if c not in (None, "00"):
            print(f"  page {p} 오류 {c}: {ERR.get(c, m)}"); continue
        for r in rs:
            if r["recallSn"] not in seen: seen.add(r["recallSn"]); rows.append(r)
        if p % 10 == 0: print(f"  {p}/{pages} 페이지, 누적 {len(rows)}건")
        time.sleep(0.2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글 안 깨지게
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"저장: {a.out} ({len(rows)}건)")


if __name__ == "__main__":
    main()
