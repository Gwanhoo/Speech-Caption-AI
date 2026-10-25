# R2-B6 RAW·조립·FINAL 원문

최종 측정 repeat 0. 반복 간 동일성은 metrics.json 참조. 모든 텍스트는 모델/조립 출력이며 정답 전사본이 아니다.

## baseline: 3초/2초

| window | 음원 구간(s) | RAW | 조립 match | 새 조각 |
|---:|---|---|---|---|
| 0 | 0–3 |  | none |  |
| 1 | 2–5 |  | none |  |
| 2 | 4–7 |  | none |  |
| 3 | 6–9 |  | none |  |
| 4 | 8–11 | 웹사이트 개발의 논리성은 | none | 웹사이트 개발의 논리성은 |
| 5 | 10–13 | 사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP | none | 사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP |
| 6 | 12–15 | 언어 PHP에서 발견된 취약점은 | boundary_exact | 에서 발견된 취약점은 |
| 7 | 14–17 | 최악점은 해커가 로그인이나 인증 없이 | none | 최악점은 해커가 로그인이나 인증 없이 |
| 8 | 16–19 | 인증 없이 외부에서 임의로 코드를 실행 | boundary_exact | 외부에서 임의로 코드를 실행 |
| 9 | 18–21 | 임의로 코드를 실행하거나 시스템을 장악할 수 있는 | boundary_exact | 하거나 시스템을 장악할 수 있는 |
| 10 | 20–23 | 시스템을 장악할 수 있는 결함으로 알려졌습니다. | token_exact | 결함으로 알려졌습니다. |
| 11 | 22–25 | 이번 사태 파급력이 큰 건 | none | 이번 사태 파급력이 큰 건 |
| 12 | 24–27 | 파급력이 큰 건 php가 인터넷 전문 | token_exact | php가 인터넷 전문 |
| 13 | 26–29 | 인터넷 전반에 광범위하게 쓰이는 핵심 | none | 인터넷 전반에 광범위하게 쓰이는 핵심 |
| 14 | 28–31 | 쓰이는 핵심 기술이기 때문입니다. | boundary_exact | 기술이기 때문입니다. |
| 15 | 30–33 | 사용된 프로그래밍 언어를 식별하며 | none | 사용된 프로그래밍 언어를 식별하며 |
| 16 | 32–35 | 프레밍 언어를 식별할 수 있는 웹사이트의 75% | supported_tail_replace | 프레밍 언어를 식별할 수 있는 웹사이트의 75% |
| 17 | 34–37 | 75%가 php를 쓰고 | none | 75%가 php를 쓰고 |
| 18 | 36–39 | 를 쓰고 | boundary_exact |  |

### FINAL

- window 11, 시간축 25s, sentence_boundary_confirmed: 웹사이트 개발의 논리성은 사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 취약점은 최악점은 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려졌습니다.
- window 15, 시간축 33s, sentence_boundary_confirmed: 이번 사태 파급력이 큰 건 php가 인터넷 전문 인터넷 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다.
- window 18, 시간축 39s, pipeline_end: 프레밍 언어를 식별할 수 있는 웹사이트의 75% 75%가 php를 쓰고

### 검토 플래그

아래는 자동 휴리스틱이 검토 대상으로 잡은 구간이다. 단어 수정·정상 병합·FINAL 간 이동도 포함하며, 실제 누락/중복 수가 아니다.

- window 11: previous_raw_ngrams_absent_from_session; absent ngrams=['결함으로알려', '는결함으로알', '로알려졌습니', '수있는결함으', '스템을장악할', '시스템을장악', '악할수있는결', '알려졌습니다', '으로알려졌습', '을장악할수있', '있는결함으로', '장악할수있는', '템을장악할수', '할수있는결함', '함으로알려졌']
- window 12: overlap_at_minimum_threshold, assembler_suspicious_deletion; absent ngrams=[]
- window 13: unmerged_fuzzy_boundary_candidate; absent ngrams=[]
- window 15: previous_raw_ngrams_absent_from_session; absent ngrams=['기때문입니다', '기술이기때문', '는핵심기술이', '술이기때문입', '심기술이기때', '쓰이는핵심기', '이기때문입니', '이는핵심기술', '핵심기술이기']
- window 16: overlap_at_minimum_threshold, assembler_suspicious_deletion, previous_raw_ngrams_absent_from_session; absent ngrams=['그래밍언어를', '된프로그래밍', '래밍언어를식', '사용된프로그', '어를식별하며', '언어를식별하', '용된프로그래']
- window 18: overlap_consumes_at_least_80_percent, assembler_suspicious_deletion; absent ngrams=[]

## A: 4초/2초

| window | 음원 구간(s) | RAW | 조립 match | 새 조각 |
|---:|---|---|---|---|
| 0 | 0–4 |  | none |  |
| 1 | 2–6 |  | none |  |
| 2 | 4–8 |  | none |  |
| 3 | 6–10 | 웹사이트 | none | 웹사이트 |
| 4 | 8–12 | 웹사이트 개발에 널리 쓰이는 프로그래밍 | none | 웹사이트 개발에 널리 쓰이는 프로그래밍 |
| 5 | 10–14 | 사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 | boundary_exact | 언어 PHP에서 발견된 |
| 6 | 12–16 | 언어 PHP에서 발견된 취약점은 해커가 로그인 | token_exact | 취약점은 해커가 로그인 |
| 7 | 14–18 | 최악점은 해커가 로그인이나 인증 없이 외부에서 | none | 최악점은 해커가 로그인이나 인증 없이 외부에서 |
| 8 | 16–20 | 인증 없이 외부에서 임의로 코드를 실행하거나 | token_exact | 임의로 코드를 실행하거나 |
| 9 | 18–22 | 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려줍니다. | token_exact | 시스템을 장악할 수 있는 결함으로 알려줍니다. |
| 10 | 20–24 | 시스템을 장악할 수 있는 결함으로 알려졌습니다. | fuzzy_replace |  |
| 11 | 22–26 | 이번 사태 파급력이 큰 건 PHP가 | none | 이번 사태 파급력이 큰 건 PHP가 |
| 12 | 24–28 | 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 | token_exact | 인터넷 전반에 광범위하게 |
| 13 | 26–30 | 인터넷 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다. | token_exact | 쓰이는 핵심 기술이기 때문입니다. |
| 14 | 28–32 | 쓰이는 핵심 기술이기 때문입니다. 사용된 프로그램은 | token_exact | 사용된 프로그램은 |
| 15 | 30–34 | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트 | none | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트 |
| 16 | 32–36 | 프레밍 언어를 식별할 수 있는 웹사이트의 75%가 php | fuzzy_replace | 75%가 php |
| 17 | 34–38 | 75%가 php를 쓰고 | boundary_exact | 를 쓰고 |
| 18 | 36–40 | 를 쓰고 | boundary_exact |  |

### FINAL

- window 11, 시간축 26s, sentence_boundary_confirmed: 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 취약점은 해커가 로그인 최악점은 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려졌습니다.
- window 14, 시간축 32s, sentence_boundary_confirmed: 이번 사태 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다.
- window 18, 시간축 40s, pipeline_end: 사용된 프로그램은 사용된 프레밍 언어를 식별할 수 있는 웹사이트의 75%가 php를 쓰고

### 검토 플래그

아래는 자동 휴리스틱이 검토 대상으로 잡은 구간이다. 단어 수정·정상 병합·FINAL 간 이동도 포함하며, 실제 누락/중복 수가 아니다.

- window 7: unmerged_fuzzy_boundary_candidate; absent ngrams=[]
- window 10: overlap_consumes_at_least_80_percent, assembler_suspicious_deletion, previous_raw_ngrams_absent_from_session; absent ngrams=['로알려줍니다', '으로알려줍니', '함으로알려줍']
- window 11: previous_raw_ngrams_absent_from_session; absent ngrams=['결함으로알려', '는결함으로알', '로알려졌습니', '수있는결함으', '스템을장악할', '시스템을장악', '악할수있는결', '알려졌습니다', '으로알려졌습', '을장악할수있', '있는결함으로', '장악할수있는', '템을장악할수', '할수있는결함', '함으로알려졌']
- window 14: previous_raw_ngrams_absent_from_session; absent ngrams=['게쓰이는핵심', '광범위하게쓰', '기때문입니다', '기술이기때문', '넷전반에광범', '는핵심기술이', '반에광범위하', '범위하게쓰이', '술이기때문입', '심기술이기때', '쓰이는핵심기', '에광범위하게', '위하게쓰이는', '이기때문입니', '이는핵심기술', '인터넷전반에', '전반에광범위', '터넷전반에광', '하게쓰이는핵', '핵심기술이기']
- window 15: unmerged_fuzzy_boundary_candidate; absent ngrams=[]
- window 16: previous_raw_ngrams_absent_from_session; absent ngrams=['그래밍언어를', '된프로그래밍', '래밍언어를식', '용된프로그래']
- window 17: overlap_at_minimum_threshold, assembler_suspicious_deletion; absent ngrams=[]
- window 18: overlap_consumes_at_least_80_percent, assembler_suspicious_deletion; absent ngrams=[]

## B: 4초/3초

| window | 음원 구간(s) | RAW | 조립 match | 새 조각 |
|---:|---|---|---|---|
| 0 | 0–4 |  | none |  |
| 1 | 3–7 |  | none |  |
| 2 | 6–10 | 웹사이트 | none | 웹사이트 |
| 3 | 9–13 | 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP. | none | 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP. |
| 4 | 12–16 | 언어 PHP에서 발견된 취약점은 해커가 로그인 | supported_sentence_revision | 언어 PHP에서 발견된 취약점은 해커가 로그인 |
| 5 | 15–19 | 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행 | boundary_exact | 이나 인증 없이 외부에서 임의로 코드를 실행 |
| 6 | 18–22 | 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려줍니다. | boundary_exact | 하거나 시스템을 장악할 수 있는 결함으로 알려줍니다. |
| 7 | 21–25 | 이번 사태 파급력이 큰 건 | none | 이번 사태 파급력이 큰 건 |
| 8 | 24–28 | 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 | token_exact | PHP가 인터넷 전반에 광범위하게 |
| 9 | 27–31 | 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다. | token_exact | 쓰이는 핵심 기술이기 때문입니다. |
| 10 | 30–34 | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트 | none | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트 |
| 11 | 33–37 | 수 있는 웹사이트의 75%가 php를 쓰고 | boundary_exact | 의 75%가 php를 쓰고 |
| 12 | 36–40 | 를 쓰고 | boundary_exact |  |

### FINAL

- window 7, 시간축 25s, sentence_boundary_confirmed: 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 취약점은 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려줍니다.
- window 10, 시간축 34s, sentence_boundary_confirmed: 이번 사태 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다.
- window 12, 시간축 40s, pipeline_end: 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트의 75%가 php를 쓰고

### 검토 플래그

아래는 자동 휴리스틱이 검토 대상으로 잡은 구간이다. 단어 수정·정상 병합·FINAL 간 이동도 포함하며, 실제 누락/중복 수가 아니다.

- window 5: overlap_at_minimum_threshold, assembler_suspicious_deletion; absent ngrams=[]
- window 7: previous_raw_ngrams_absent_from_session; absent ngrams=['거나시스템을', '결함으로알려', '나시스템을장', '는결함으로알', '드를실행하거', '로알려줍니다', '로코드를실행', '를실행하거나', '수있는결함으', '스템을장악할', '시스템을장악', '실행하거나시', '악할수있는결', '으로알려줍니', '을장악할수있', '의로코드를실', '임의로코드를', '있는결함으로', '장악할수있는', '코드를실행하', '템을장악할수', '하거나시스템', '할수있는결함', '함으로알려줍', '행하거나시스']
- window 8: overlap_at_minimum_threshold, assembler_suspicious_deletion; absent ngrams=[]
- window 10: previous_raw_ngrams_absent_from_session; absent ngrams=['게쓰이는핵심', '광범위하게쓰', '기때문입니다', '기술이기때문', '는핵심기술이', '반에광범위하', '범위하게쓰이', '술이기때문입', '심기술이기때', '쓰이는핵심기', '에광범위하게', '위하게쓰이는', '이기때문입니', '이는핵심기술', '전반에광범위', '하게쓰이는핵', '핵심기술이기']
- window 12: overlap_consumes_at_least_80_percent, assembler_suspicious_deletion; absent ngrams=[]

## C: 5초/3초

| window | 음원 구간(s) | RAW | 조립 match | 새 조각 |
|---:|---|---|---|---|
| 0 | 0–5 |  | none |  |
| 1 | 3–8 |  | none |  |
| 2 | 6–11 | 웹사이트 개발의 논리 | none | 웹사이트 개발의 논리 |
| 3 | 9–14 | 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 | none | 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 |
| 4 | 12–17 | 언어 PHP에서 발견된 취약점은 해커가 로그인이나 인증 없이 | token_exact | 취약점은 해커가 로그인이나 인증 없이 |
| 5 | 15–20 | 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행하거나 | token_exact | 외부에서 임의로 코드를 실행하거나 |
| 6 | 18–23 | 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려졌습니다. | token_exact | 시스템을 장악할 수 있는 결함으로 알려졌습니다. |
| 7 | 21–26 | 이번 사태 파급력이 큰 건 PHP가 | none | 이번 사태 파급력이 큰 건 PHP가 |
| 8 | 24–29 | 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 쓰이는 핵심 | token_exact | 인터넷 전반에 광범위하게 쓰이는 핵심 |
| 9 | 27–32 | 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다. | token_exact | 기술이기 때문입니다. |
| 10 | 30–35 | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트의 75% | none | 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트의 75% |
| 11 | 33–38 | 수 있는 웹사이트의 75%가 php를 쓰고 | normalized_exact | %가 php를 쓰고 |

### FINAL

- window 7, 시간축 26s, sentence_boundary_confirmed: 웹사이트 개발의 논리 웹사이트 개발에 널리 쓰이는 프로그래밍 언어 PHP에서 발견된 취약점은 해커가 로그인이나 인증 없이 외부에서 임의로 코드를 실행하거나 시스템을 장악할 수 있는 결함으로 알려졌습니다.
- window 10, 시간축 35s, sentence_boundary_confirmed: 이번 사태 파급력이 큰 건 PHP가 인터넷 전반에 광범위하게 쓰이는 핵심 기술이기 때문입니다.
- window 11, 시간축 38s, pipeline_end: 사용된 프로그래밍 언어를 식별할 수 있는 웹사이트의 75%가 php를 쓰고

### 검토 플래그

아래는 자동 휴리스틱이 검토 대상으로 잡은 구간이다. 단어 수정·정상 병합·FINAL 간 이동도 포함하며, 실제 누락/중복 수가 아니다.

- window 7: previous_raw_ngrams_absent_from_session; absent ngrams=['거나시스템을', '결함으로알려', '나시스템을장', '는결함으로알', '드를실행하거', '로알려졌습니', '로코드를실행', '를실행하거나', '수있는결함으', '스템을장악할', '시스템을장악', '실행하거나시', '악할수있는결', '알려졌습니다', '으로알려졌습', '을장악할수있', '의로코드를실', '임의로코드를', '있는결함으로', '장악할수있는', '코드를실행하', '템을장악할수', '하거나시스템', '할수있는결함', '함으로알려졌', '행하거나시스']
- window 10: previous_raw_ngrams_absent_from_session; absent ngrams=['게쓰이는핵심', '광범위하게쓰', '기때문입니다', '기술이기때문', '는핵심기술이', '반에광범위하', '범위하게쓰이', '술이기때문입', '심기술이기때', '쓰이는핵심기', '에광범위하게', '위하게쓰이는', '이기때문입니', '이는핵심기술', '전반에광범위', '하게쓰이는핵', '핵심기술이기']

