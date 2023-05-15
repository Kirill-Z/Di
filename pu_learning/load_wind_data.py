from load_fog_data import Data


class WindData(Data):
    def __init__(self):
        super().__init__()
        self.dir_path = "/home/kirill/PycharmProjects/Di/wind_data"
        self.clear_data_path = "/home/kirill/PycharmProjects/Di/clear_wind_data"

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

a = WindData()()
