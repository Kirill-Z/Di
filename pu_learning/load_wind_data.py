from load_data import Data


class WindData(Data):
    def __init__(self, garbage):
        super().__init__(garbage)
        self.dir_path = "/home/kirill/PycharmProjects/Di/wind_data"
        self.clear_data_path = "/home/kirill/PycharmProjects/Di/clear_wind_data"

    def get_data(self):
        true_data, false_data = self.get_raw_data()
        return self.conver_str_to_num(true_data, int), self.conver_str_to_num(
            false_data, int
        )

    def del_garbage(self, data):
        len_data = len(data)
        i = 0
        while i < len_data:
            for j in range(len(data[i])):
                if data[i][j] == "9999":
                    data.pop(i)
                    len_data -= 1
                    i -= 1
                    break
            i += 1
        return data
