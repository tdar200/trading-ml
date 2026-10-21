"""Versioned research feature contracts. No feature claims local predictiveness."""
from fundamentals.providers import SEC_TAGS
from fundamentals.features import EVIDENCE

FORMULAS={
 'fcf':'ocf - positive cash capex',
 'ebitda_proxy':'operating_income + depreciation; proxy, not reported EBITDA',
 'debt':'short_debt + long_debt (both required)', 'net_debt':'debt - cash',
 'gross_margin':'gross_profit / revenue','operating_margin':'operating_income / revenue',
 'net_margin':'net_income / revenue','fcf_margin':'fcf / revenue','ebitda_margin':'ebitda_proxy / revenue',
 'roe':'net_income / mean(current equity, prior comparable-year equity)',
 'roa':'net_income / mean(current assets, prior comparable-year assets)',
 'gross_profitability':'gross_profit / assets',
 'roic':'operating_income * (1 - tax_expense/pretax_income) / average invested capital; invested capital = debt + equity - cash; tax rate must be in [0,1]',
 'current_ratio':'current_assets / current_liabilities',
 'quick_ratio':'(current_assets - inventory - prepaids) / current_liabilities; missing prepaids are not zero',
 'debt_equity':'debt / positive equity','net_debt_ebitda':'net_debt / positive EBITDA proxy',
 'interest_coverage':'operating_income / positive interest_expense','cash_debt':'cash / positive debt',
 'ocf_net_income':'ocf / positive net_income','fcf_net_income':'fcf / positive net_income',
 'accrual_ratio':'(net_income - ocf) / average assets; cash-flow accrual proxy',
 'revenue_acceleration':'current revenue YoY minus prior comparable revenue YoY',
 'eps_acceleration':'current EPS YoY minus prior EPS YoY','fcf_acceleration':'current FCF YoY minus prior FCF YoY',
 'receivables_vs_sales':'receivables YoY minus revenue YoY','inventory_vs_sales':'inventory YoY minus revenue YoY',
 'gross_margin_change':'current gross margin minus prior comparable gross margin',
 'operating_margin_change':'current operating margin minus prior comparable operating margin',
 'debt_change':'current debt minus prior comparable debt','cash_change':'current cash minus prior comparable cash',
 'asset_turnover':'annual revenue / average assets','inventory_turnover':'annual COGS / average inventory',
 'rd_sales':'R&D / revenue','capex_sales':'cash capex / revenue','buybacks_ocf':'buyback cash / OCF',
 'dividends_ocf':'dividend cash / OCF','acquisitions_assets':'acquisition cash / assets','issuance_equity':'issuance proceeds / positive equity',
 'receivable_days':'365 * average receivables / annual revenue',
 'inventory_days':'365 * average inventory / annual COGS','payable_days':'365 * average payables / annual COGS',
 'cash_conversion_cycle':'receivable_days + inventory_days - payable_days',
 'incremental_roic':'change NOPAT / positive change invested capital',
 'gross_margin_stability':'sample standard deviation of three known annual gross margins; lower means more stable',
 'earnings_stability':'sample standard deviation of three annual net incomes / absolute mean; undefined at zero mean',
 'revenue_cagr_3y':'(current revenue / positive revenue three fiscal years earlier)^(1/3)-1',
 'revenue_cagr_5y':'(current revenue / positive revenue five fiscal years earlier)^(1/5)-1',
}
for metric in ('revenue','eps','operating_income','receivables','inventory','diluted_shares','cash','assets','fcf','ebitda'):
    FORMULAS[metric+'_yoy']=f'current {metric} / positive comparable prior-year {metric} - 1'
for metric in ('revenue','eps','ocf','operating_income'):
    FORMULAS[metric+'_qoq']=f'latest discrete-quarter {metric} / positive immediately prior quarter - 1; YTD differenced before comparison'

RATIONALES={
 'quality':'Cash-backed profitability and capital efficiency may distinguish persistent business economics from price strength.',
 'growth':'Changes in the business may convey information beyond the current level; high growth alone can be expensive.',
 'health':'Liquidity and debt service affect distress exposure; the same leverage ratio has different meaning by industry.',
 'earnings_quality':'Divergence between earnings, working capital and operating cash may reveal transitory earnings; not an allegation of fraud.',
 'allocation':'Investment, dilution and distributions affect future per-share cash flows; direction is conditional on reinvestment returns.',
 'value':'Price paid relative to economic claims affects prospective return; cheapness may reflect distress.',
}


def feature_dictionary():
    from fundamentals.features import accounting_features
    _,dimensions,_=accounting_features({})
    for k in FORMULAS:
        dimensions.setdefault(k,'growth')
    return {name:{'formula':formula,'dimension':dimensions[name],'economic_rationale':RATIONALES[dimensions[name]],
        'source':'SEC accession-tagged US-GAAP facts or validated provider-neutral dated records',
        'availability':'max publication availability of all dependency versions; never period_end',
        'historical_availability':'SEC structured reporting generally from 2009; tag/issuer coverage varies',
        'normalization':'raw accounting ratio/change first; PIT sector percentile or train-only standardization for models',
        'missing_rule':'None for absent inputs, mismatched currency/duration, or invalid denominator; no zero imputation',
        'validation':'deterministic arithmetic/PIT fixtures; purged 1/5/20/60/120-session IC and cost-aware temporal ablation when coverage exists',
        'evidence':EVIDENCE[dimensions[name]],'local_predictive_usefulness':'NOT_ESTABLISHED','production_weight':None}
        for name,formula in FORMULAS.items()}

RAW_DICTIONARY={metric:{'sec_tags_priority':tags,'units':'source units preserved; currencies never implicitly combined',
                       'timestamp':'filing acceptance +5min; if unavailable, following New York calendar day',
                       'revision':'SEC accession','normalization':'none before accounting transformations'} for metric,tags in SEC_TAGS.items()}
