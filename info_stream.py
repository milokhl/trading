from googlefinance import getQuotes
import json

def main():
	quotes = getQuotes('GOOG')
	print quotes


if __name__ == '__main__':
	main()