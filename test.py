import argparse
import json
import os
from datetime import datetime

import pandas as pd
from Levenshtein import ratio


def parse_args(desc=''):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-p',
        '--pred',
        type=str,
        required=True,
        help='Input path to the preds'
    )
    parser.add_argument(
        '-t',
        '--true',
        type=str,
        required=True,
        help='Input path to the true'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        default='',
        help='Output path for saving result (metrics, errors)'
    )
    parser.add_argument(
        '-T',
        '--threshold',
        type=float,
        default=0.9,
        help='Threshold for Levenshtein-ratio (default: 0.9)'
    )
    return parser.parse_args()

def evaluate(y_true, y_pred, threshold=0.9):
    categories = ['title', 'authors_name', 'authors_aff', 'abstract', 'keywords', 'sections_title', 'sections_text', 'references', 'ref_year', 'refs_authors', 'tables', 'table_captions']
    counters = {category: {"tp": 0, "fp": 0, "fn": 0} for category in categories}
    errors = {category: [] for category in categories}

    for idx, y in enumerate(y_true):
        # title
        title_err_fl = None
        if y_pred[idx]['title'] == None and y != None:
            counters['title']['fn'] += 1
            title_err_fl = 'False Negative (none)'
        elif ratio(y['title'], y_pred[idx]['title'], processor=lambda s: s.lower()) >= threshold:
            counters['title']['tp'] += 1
        else:
            counters['title']['fp'] += 1
            title_err_fl = 'False Positive (incorrect extraction)'

        if title_err_fl != None:
            errors['title'].append({
                'pred': y_pred[idx]['title'],
                'true': y['title'],
                'type': title_err_fl
            })

        # authors
        authors_matched = []
        authors_tp = 0
        for a in y['authors']:
            authors_err_fl = None
            best_l_ratio = 0
            best_jdx = -1

            for jdx, a_pred in enumerate(y_pred[idx]['authors']):
                if jdx not in authors_matched:
                    l_ratio = ratio(a['name'], a_pred['name'], processor=lambda s: s.lower())
                    if l_ratio > best_l_ratio:
                        best_l_ratio = l_ratio
                        best_jdx = jdx

            if best_l_ratio >= threshold:
                authors_tp += 1
                authors_matched.append(best_jdx)
                # authors_aff
                aff_matched = []
                aff_tp = 0
                for aff in a['affiliation']:
                    aff_err_fl = None
                    best_l_ratio = 0
                    best_kdx = -1

                    for kdx, aff_pred in enumerate(y_pred[idx]['authors'][best_jdx]['affiliation']):
                        if kdx not in aff_matched:
                            l_ratio = ratio(aff, aff_pred, processor=lambda s: s.lower())
                            if l_ratio > best_l_ratio:
                                best_l_ratio = l_ratio
                                best_kdx = kdx

                    if best_l_ratio >= threshold:
                        aff_tp += 1
                        aff_matched.append(best_kdx)
                    elif best_l_ratio >= threshold//2:
                        aff_err_fl = 'False Positive (incorrect extraction)'
                    else:
                        aff_err_fl = 'False Negative (no suitable value)'
                    if aff_err_fl != None:
                        errors['authors_aff'].append({
                            'pred': y_pred[idx]['authors'][best_jdx]['affiliation'][best_kdx],
                            'true': aff,
                            'type': aff_err_fl
                        })

                counters['authors_aff']['tp'] += aff_tp
                counters['authors_aff']['fp'] += len(y_pred[idx]['authors'][best_jdx]['affiliation']) - aff_tp
                counters['authors_aff']['fn'] += len(a['affiliation']) - aff_tp
            elif best_l_ratio >= threshold//2:
                authors_err_fl = 'False Positive (incorrect extraction)'
            else:
                authors_err_fl = 'False Negative (no suitable value)'

            if authors_err_fl != None:
                errors['authors_name'].append({
                    'pred': y_pred[idx]['authors'][best_jdx],
                    'true': a,
                    'type': authors_err_fl
                })

        counters['authors_name']['tp'] += authors_tp
        counters['authors_name']['fp'] += len(y_pred[idx]['authors']) - authors_tp
        counters['authors_name']['fn'] += len(y['authors']) - authors_tp

        # abstract
        abstract_err_fl = None
        if y_pred[idx]['abstract'] == None and y['abstract'] != None:
            counters['abstract']['fn'] += 1
            abstract_err_fl = 'False Negative (none)'
        elif ratio(y['abstract'], y_pred[idx]['abstract'], processor=lambda s: s.lower()) >= threshold:
            counters['abstract']['tp'] += 1
        else:
            counters['abstract']['fp'] += 1
            abstract_err_fl = 'False Positive (incorrect extraction)'

        if abstract_err_fl != None:
            errors['abstract'].append({
                'pred': y_pred[idx]['abstract'],
                'true': y['abstract'],
                'type': abstract_err_fl
            })

        # keywords
        keywords_err_fl = None
        if y['keywords'] == None:
            if y_pred[idx]['keywords'] == None:
                counters['keywords']['tp'] += 1
            else:
                counters['keywords']['fp'] += 1
                keywords_err_fl = 'False Positive (incorrect extraction)'
                errors['keywords'].append({
                    'pred': y_pred[idx]['keywords'],
                    'true': None,
                    'type': keywords_err_fl
                })
        else:
            kw_matched = []
            kw_tp = 0
            for kw in y['keywords']:
                keywords_err_fl = None
                best_l_ratio = 0
                best_jdx = -1

                for jdx, kw_pred in enumerate(y_pred[idx]['keywords']):
                    if jdx not in kw_matched:
                        l_ratio = ratio(kw, kw_pred, processor=lambda s: s.lower())
                        if l_ratio > best_l_ratio:
                            best_l_ratio = l_ratio
                            best_jdx = jdx

                if best_l_ratio >= threshold:
                    kw_tp += 1
                    kw_matched.append(best_jdx)
                elif best_l_ratio >= threshold//2:
                    keywords_err_fl = 'False Positive (incorrect extraction)'
                else:
                    keywords_err_fl = 'False Negative (no suitable value)'

                if keywords_err_fl != None:
                    errors['keywords'].append({
                        'pred': y_pred[idx]['keywords'][best_jdx],
                        'true': kw,
                        'type': keywords_err_fl
                    })

            counters['keywords']['tp'] += kw_tp
            counters['keywords']['fp'] += len(y_pred[idx]['keywords']) - kw_tp
            counters['keywords']['fn'] += len(y['keywords']) - kw_tp

        # sections_title
        stitle_matched = []
        s_tp = 0
        for sec in y['sections']:
            sec_title_err_fl = None
            best_l_ratio = 0
            best_jdx = -1

            for jdx, s_pred in enumerate(y_pred[idx]['sections']):
                if jdx not in stitle_matched:
                    l_ratio = ratio(sec['title'], s_pred['title'], processor=lambda s: s.lower())
                    if l_ratio > best_l_ratio:
                        best_l_ratio = l_ratio
                        best_jdx = jdx

            if best_l_ratio >= threshold:
                s_tp += 1
                stitle_matched.append(best_jdx)
            elif best_l_ratio >= threshold//2:
                sec_title_err_fl = 'False Positive (incorrect extraction)'
            else:
                sec_title_err_fl = 'False Negative (no suitable value)'

            if sec_title_err_fl != None:
                errors['sections_title'].append({
                    'pred': y_pred[idx]['sections'][best_jdx],
                    'true': sec,
                    'type': sec_title_err_fl
                })

        counters['sections_title']['tp'] += s_tp
        counters['sections_title']['fp'] += len(y_pred[idx]['sections']) - s_tp
        counters['sections_title']['fn'] += len(y['sections']) - s_tp

        # sections_text
        stext_matched = []
        stext_tp = 0
        for sec in y['sections']:
            sec_text_err_fl = None
            best_l_ratio = 0
            best_jdx = -1

            for jdx, s_pred in enumerate(y_pred[idx]['sections']):
                if jdx not in stext_matched:
                    l_ratio = ratio(sec['text'], s_pred['text'], processor=lambda s: s.lower())
                    if l_ratio > best_l_ratio:
                        best_l_ratio = l_ratio
                        best_jdx = jdx

            if best_l_ratio >= threshold:
                stext_tp += 1
                stext_matched.append(best_jdx)
            elif best_l_ratio >= threshold//2:
                sec_text_err_fl = 'False Positive (incorrect extraction)'
            else:
                sec_text_err_fl = 'False Negative (no suitable value)'

            if sec_text_err_fl != None:
                errors['sections_text'].append({
                    'pred': y_pred[idx]['sections'][best_jdx],
                    'true': sec,
                    'type': sec_text_err_fl
                })

        counters['sections_text']['tp'] += stext_tp
        counters['sections_text']['fp'] += len(y_pred[idx]['sections']) - stext_tp
        counters['sections_text']['fn'] += len(y['sections']) - stext_tp

        # references
        rtext_matched = []
        rtext_tp = 0
        for r in y['references']:
            ref_err_fl = None
            ref_best_l_ratio = 0
            best_jdx = -1

            for jdx, r_pred in enumerate(y_pred[idx]['references']):
                if jdx not in rtext_matched:
                    l_ratio = ratio(r['text'], r_pred['text'], processor=lambda s: s.lower())
                    if l_ratio > ref_best_l_ratio:
                        ref_best_l_ratio = l_ratio
                        best_jdx = jdx

            if ref_best_l_ratio >= threshold:
                rtext_tp += 1
                stext_matched.append(best_jdx)
                # year
                year_err_fl = None
                if y_pred[idx]['references'][best_jdx]['year'] == r['year']:
                    counters['ref_year']['tp'] += 1
                elif y_pred[idx]['references'][best_jdx]['year'] == None:
                    counters['ref_year']['fn'] += 1
                    year_err_fl = 'False Negative (none)'
                else:
                    counters['ref_year']['fp'] += 1
                    year_err_fl = 'False Positive (incorrect extraction)'

                if year_err_fl != None:
                    errors['ref_year'].append({
                        'pred': y_pred[idx]['references'][best_jdx]['year'],
                        'true': r['year'],
                        'type': year_err_fl
                    })

                # refs_authors
                ref_a_matched = []
                raut_tp = 0
                for a in r['authors']:
                    ref_authors_err_fl = None
                    best_l_ratio = 0
                    best_kdx = -1

                    for kdx, a_pred in enumerate(y_pred[idx]['references'][best_jdx]['authors']):
                        if kdx not in ref_a_matched:
                            l_ratio = ratio(a, a_pred, processor=lambda s: s.lower())
                            if l_ratio > best_l_ratio:
                                best_l_ratio = l_ratio
                                best_kdx = kdx

                    if best_l_ratio >= threshold:
                        raut_tp += 1
                        ref_a_matched.append(best_kdx)
                    elif best_l_ratio >= threshold//2:
                        ref_authors_err_fl = 'False Positive (incorrect extraction)'
                    else:
                        ref_authors_err_fl = 'False Negative (no suitable value)'

                    if ref_authors_err_fl != None:
                        errors['refs_authors'].append({
                            'pred': y_pred[idx]['references'][best_jdx]['authors'][best_kdx],
                            'true': a,
                            'type': ref_authors_err_fl
                        })

                counters['refs_authors']['tp'] += raut_tp
                counters['refs_authors']['fp'] += len(y_pred[idx]['references'][best_jdx]['authors']) - raut_tp
                counters['refs_authors']['fn'] += len(r['authors']) - raut_tp
            elif ref_best_l_ratio >= threshold//2:
                ref_err_fl = 'False Positive (incorrect extraction)'
            else:
                ref_err_fl = 'False Negative (no suitable value)'

            if ref_err_fl != None:
                errors['references'].append({
                    'pred': y_pred[idx]['references'][best_jdx],
                    'true': r,
                    'type': ref_err_fl
                })

        counters['references']['tp'] += rtext_tp
        counters['references']['fp'] += len(y_pred[idx]['references']) - rtext_tp
        counters['references']['fn'] += len(y['references']) - rtext_tp

        # tables (table_body)
        tables_matched = []
        tables_tp = 0
        for t in y['tables']:
            table_best_l_ratio = 0
            best_jdx = -1
            table_body_err_fl = None

            for jdx, t_pred in enumerate(y_pred[idx]['tables']):
                if jdx not in tables_matched:
                    l_ratio = ratio(t['table_body'], t_pred['table_body'], processor=lambda s: s.lower())
                    if l_ratio > table_best_l_ratio:
                        table_best_l_ratio = l_ratio
                        best_jdx = jdx

            if table_best_l_ratio >= threshold:
                tables_tp += 1
                tables_matched.append(best_jdx)
                # tables_caption
                tcap_matched = []
                tcap_tp = 0
                for cap in t['caption']:
                    table_caption_err_fl = None
                    best_l_ratio = 0
                    best_kdx = -1

                    for kdx, cap_pred in enumerate(y_pred[idx]['tables'][best_jdx]['caption']):
                        if kdx not in tcap_matched:
                            l_ratio = ratio(cap, cap_pred, processor=lambda s: s.lower())
                            if l_ratio > best_l_ratio:
                                best_l_ratio = l_ratio
                                best_kdx = kdx

                    if best_l_ratio >= threshold:
                        tcap_tp += 1
                        tcap_matched.append(best_kdx)
                    elif best_l_ratio >= threshold//2:
                        table_caption_err_fl = 'False Positive (incorrect extraction)'
                    else:
                        table_caption_err_fl = 'False Negative (no suitable value)'

                    if table_caption_err_fl != None:
                        errors['table_captions'].append({
                            'pred': y_pred[idx]['tables'][best_jdx]['caption'][best_kdx],
                            'true': cap,
                            'type': table_caption_err_fl
                        })

                counters['table_captions']['tp'] += tcap_tp
                counters['table_captions']['fp'] += len(y_pred[idx]['tables'][best_jdx]['caption']) - tcap_tp
                counters['table_captions']['fn'] += len(t['caption']) - tcap_tp
            elif table_best_l_ratio >= threshold//2:
                table_body_err_fl = 'False Positive (incorrect extraction)'
            else:
                table_body_err_fl = 'False Negative (no suitable value)'

            if table_body_err_fl != None:
                errors['tables'].append({
                    'pred': y_pred[idx]['tables'][best_jdx],
                    'true': t,
                    'type': table_body_err_fl
                })

        counters['tables']['tp'] += tables_tp
        counters['tables']['fp'] += len(y_pred[idx]['tables']) - tables_tp
        counters['tables']['fn'] += len(y['tables']) - tables_tp

        error_data = []
        for category in errors.keys():
            for e in errors[category]:
                e['name'] = category
            error_data += errors[category]

    return counters, pd.DataFrame(data=error_data)

if __name__ == '__main__':
    args = parse_args()
    print(os.getcwd())
    path_to_preds = [os.path.join(args.pred, p) for p in os.listdir(path=args.pred)]
    path_to_true = [os.path.join(args.true, p) for p in os.listdir(path=args.true)]
    output_path = args.output_path
    threshold = args.threshold

    y_true = []
    y_pred = []

    for p in path_to_true:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            y_true.append(data)

    for p in path_to_preds:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            y_pred.append(data)

    metrics, errors = evaluate(y_true, y_pred, threshold)
    errors.to_csv(f'{output_path}/errors_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv', index=False)

    for cat in metrics:
        print('-', cat, '-'*(29 - len(cat)))
        precision = metrics[cat]['tp'] / (metrics[cat]['tp'] + metrics[cat]['fp'])
        recall = metrics[cat]['tp'] / (metrics[cat]['tp'] + metrics[cat]['fn'])
        f1 = 2 * (precision*recall) / (precision+recall)
        print('precision:', str(round(precision, 2)).rjust(4))
        print('recall:', str(round(recall, 2)).rjust(7))
        print('f1:', str(round(f1, 2)).rjust(11))
        print("-"*32)
        print()
