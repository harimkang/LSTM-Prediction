%	Date : 2019.12.03
%	Programmer : Harim Kang
%	Description : Extract data that only the daily average jeans price from the online
%	collection price data from that source.

%	Data Cleaning : 2015.01~2019.10 Korean jeans online sale average
%	Data Ref : https://www.data.go.kr/dataset/15004449/fileData.do
%	The dataset is not provided by GitHub, only the output. : jean_sales.xlsx
%	function : extract_jean_sale_average

function [dataset] = extract_jean_sale_average(year)
	dataset = table();
	month = ['01'; '02'; '03'; '04'; '05'; '06'; '07'; '08'; '09'; '10'; '11'; '12'];
	for i=1 : length(month)
		path = sprintf('%s\\%s%s\\*.csv', year, year, month(i,:));
		datafiles = dir(path);
		numoffiles = length(datafiles);
		for j=1 : numoffiles
			datafilename = [datafiles(j).folder '\' datafiles(j).name];
			T = readtable(datafilename);
			if(isempty(T))
				continue
			else
				try
					T = T(:,{'collect_day','pum_name','sales_price'});
				catch
					fprintf('Error : %s - %d',month(i,2),j);
					continue
				end
				T.collect_day = datetime(T.collect_day);
				today = T.collect_day(1);
				T.pum_name = categorical(T.pum_name);
				n_T = table();
				c_index = T.pum_name == '청바지';
				A = T(c_index,:);
				cell_T = {today, '청바지', mean(A.sales_price)};
				n_T = [n_T;cell_T];
				n_T.Properties.VariableNames = T.Properties.VariableNames;
				if(isempty(dataset))
					dataset = n_T;
				else
					dataset = [dataset;n_T];
				end
			end
			fprintf('%d',j);
			clearvars -except dataset datafiles numoffiles year month i j
		end
		fprintf('%d\n',i);
	end
end

year = ['2015';'2016';'2017';'2018';'2019'];
jean_sales = table();
for i=1 : length(year)
	tt = extract_jean_sale_average(year(i,:));
	jean_sales = [jean_sales; tt];