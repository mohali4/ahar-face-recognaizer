import multiprocessing as mp

class baseProcess(mp.Process):
    def __init__(self,*args,**wargs):
        super().__init__(*args,**wargs)
        # self.start()

class basePPool :
    all:list[baseProcess] = []
    num = 5
    processModel = baseProcess
    def add(self, process:baseProcess):
        self.all.append(process)
    @property
    def len(self):
        return len(self.all)
    def init(self,*args):
        if self.len < self.num:
            for _ in range(self.num-self.len):
                self.new(*args)
    def __init__(self, num=None):
        if num :
            self.num = num
    def new(self,*args,**wargs):
        self.add(
            self.make_process(*args, **wargs)
        )
    def make_process(self,*args, **wargs):
        return  self.processModel(
                    target=self.target,
                    args=self._get_args(args),
                    wargs=self._get_wargs(wargs)
            )

    def _get_args(self,_input=None):
        return tuple()
    def _get_wargs(self,_input=None):
        return dict()
    def _make_shared_values(self):
        return tuple()
    def join(self):
        for process in self.all :
            process.join()
    def target(self):
        raise
    def start(self):
        for p in self.all:
            p.start()