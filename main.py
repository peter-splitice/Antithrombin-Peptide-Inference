## Main function for implemention.

from inference import *

if __name__ == '__main__':
    pd.set_option('display.max_rows',None)
    inference('combined_hits.csv', ensemble=True)
    inference('APEADQTTPEEKPAEPEPVA', ensemble=True)
    inference('APEADQTTPEEKPAEPEPVA', ensemble=False)
    inference('QSPLPERQE', ensemble=True)
    inference('QSPLPERQE', ensemble=False)
    inference('HTLGYINDNEEGPR', ensemble=True)
    inference('HTLGYINDNEEGPR', ensemble=False)
    inference('smallhits.fasta', ensemble=True)
    inference('smallhits.fasta', ensemble=False)
    inference('longhits.fasta', ensemble=True)
    inference('longhits.fasta', ensemble=False)