from functools import reduce
import operator

class RptSetup(object):
    """
    Configuration file for querying HLS report content.
  
    ...
  
    Attributes
    ----------
        Different attributes in the report file.

    Methods
    ----------
    _lookup(self, keys)
        Look up the value stored under the list of keys.

    eval_members(self)
        Populate attributes with respective values. 
    """
    def __init__(self, profile, prod_name):
        """
        Parameters
        ----------
        profile: dict
            Dictionary representation of the report file.

        prod_name: str
            Name of the target device.
        """
        self.profile = profile
        self.prod_name = prod_name
        self.version = ["ReportVersion", "Version"]

        # Units
        self.assignment_unit = ["UserAssignments", "unit"]
        self._perf_est = ["PerformanceEstimates"]
        self._overall_latency = self._perf_est + ["SummaryOfOverallLatency"]
        self.performance_unit = self._overall_latency + ["unit"]

        # User assignments
        self._user_assignments = ["UserAssignments"]
        self.prod_family = self._user_assignments + ["ProductFamily"]
        self.target_device = self._user_assignments + ["Part"]
        self.top_model_name = self._user_assignments + ["TopModelName"]
        self.target_cp = self._user_assignments + ["TargetClockPeriod"]

        # Area estimates
        self._area_est = ["AreaEstimates"]
        self.est_resources = self._area_est + ["Resources"]
        self.avail_resources = self._area_est + ["AvailableResources"]

        # Performance estimates
        self.estimated_cp = self._perf_est + ["SummaryOfTimingAnalysis", "EstimatedClockPeriod"]
        self.min_latency = self._overall_latency + ["Best-caseLatency"]
        self.max_latency = self._overall_latency + ["Worst-caseLatency"]
        self.min_interval = self._overall_latency + ["Interval-min"]
        self.max_interval = self._overall_latency + ["Interval-max"]

        # Loop latency
        self.loop_latency = self._perf_est + ["SummaryOfLoopLatency"]

    def _lookup(self, keys):
        """Lookup the content stored under the list of keys.

        Parameters
        ----------
        keys : lst
            List of key names. 

        Returns
        ----------
        dict OR str
            Value that corresponds to such lists of keys indexed to.

        Raises
        ----------
        KeyError
            Attempting to index with invalid key.
        """
        try:
            return reduce(operator.getitem, keys, self.profile)
        except KeyError:
            print("Invalid key")
            raise

    def eval_members(self):
        """Initialize each attribute to appropriate values.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        fields = vars(self)
        for k, v in fields.items():
            if k[0] != "_" and isinstance(v, list):
                val = getattr(self, k)
                val = self._lookup(val)
                setattr(self, k, val)

