"""Independent logical-clock checks; no runtime or speed measurements."""
import builtins
import json
import unittest
from unittest.mock import patch

from bench.workbench_refiner_timing import RefinerStageProfiler, PHASES


class IndependentRefinerTimingChecks(unittest.TestCase):
    def test_seven_disjoint_phases_and_inclusive_total_are_not_additive(self):
        now = [10.]
        p = RefinerStageProfiler(clock=lambda: now[0], include_total=True)
        with p.trace({'rng_seed': 3}) as trace:
            for i, name in enumerate(PHASES):
                now[0] += .25  # deliberately unassigned gaps
                with trace.stage(name):
                    now[0] += (i+1)/100
        s = p.snapshot()
        row = s['traces'][0]
        self.assertEqual(row['outcome'], 'completed')
        self.assertEqual([x['name'] for x in row['phases']], list(PHASES))
        for i, phase in enumerate(row['phases']):
            self.assertAlmostEqual(phase['elapsed_seconds'], (i+1)/100)
        self.assertAlmostEqual(sum(x['elapsed_seconds'] for x in row['phases']), .28)
        self.assertAlmostEqual(row['elapsed_seconds'], 7*.25+.28)
        self.assertIn('NOT_GPU', s['mode'])
        self.assertFalse(s['perturbation'])

    def test_explicit_sync_cost_is_labelled_and_capacity_stops_observation(self):
        now = [5.]
        events = []
        def clock():
            events.append('clock'); return now[0]
        def sync():
            events.append('sync'); now[0] += .002
        p = RefinerStageProfiler(max_traces=1, clock=clock, synchronize=sync,
                                 device='caller-declared-fake-device')
        with p.trace() as trace:
            with trace.stage('network_with_input_transfers'):
                events.append('body'); now[0] += .003
        first = list(events)
        with p.trace() as trace:
            with trace.stage('network_with_input_transfers'):
                events.append('dropped-body')
        self.assertEqual(first, ['sync', 'clock', 'body', 'sync', 'clock'])
        self.assertEqual(events, first+['dropped-body'])
        s = p.snapshot()
        self.assertEqual(s['dropped_traces'], 1)
        self.assertEqual(len(s['traces']), 1)
        self.assertTrue(s['perturbation'])
        self.assertTrue(s['synchronization_overhead_included'])
        self.assertAlmostEqual(s['traces'][0]['phases'][0]['elapsed_seconds'], .005)

    def test_runtime_failure_is_preserved_with_partial_observations(self):
        values = iter([2., 2.125])
        p = RefinerStageProfiler(clock=lambda: next(values))
        original = RuntimeError('manufactured failure, no inference')
        with self.assertRaises(RuntimeError) as caught:
            with p.trace({'rng_seed': 0}) as trace:
                with trace.stage('pnp_solver'):
                    raise original
        self.assertIs(caught.exception, original)
        row = p.snapshot()['traces'][0]
        self.assertEqual(row['outcome'], 'failed')
        self.assertEqual(row['phases'][0]['outcome'], 'failed')
        self.assertEqual(row['phases'][0]['error'], 'RuntimeError')
        self.assertEqual(row['phases'][0]['elapsed_seconds'], .125)

    def test_clock_failure_is_not_a_zero_duration_success(self):
        values = iter([1., float('nan')])
        p = RefinerStageProfiler(clock=lambda: next(values))
        with self.assertRaises(ValueError):
            with p.trace() as trace:
                with trace.stage('correspondences'):
                    pass
        row = p.snapshot()['traces'][0]
        self.assertEqual(row['outcome'], 'failed')
        self.assertIsNone(row['phases'][0]['elapsed_seconds'])
        self.assertEqual(row['phases'][0]['outcome'], 'failed')
        json.dumps(p.snapshot(), allow_nan=False)

    def test_detached_context_and_snapshot_without_runtime_imports(self):
        context = {'run': {'id': 'test'}}
        metadata = {'rng_seed': 2, 'operator': ['unit-control']}
        real = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'torchvision', 'sam2', 'gotrack'}:
                raise AssertionError('unexpected runtime import: '+name)
            return real(name, *args, **kwargs)
        with patch('builtins.__import__', guard):
            p = RefinerStageProfiler(context=context)
            with p.trace(metadata):
                pass
            context['run']['id'] = 'changed'
            metadata['operator'].append('changed')
            s = p.snapshot()
        self.assertEqual(s['context'], {'run': {'id': 'test'}})
        self.assertEqual(s['traces'][0]['metadata']['operator'], ['unit-control'])
        s['traces'].clear()
        self.assertEqual(len(p.snapshot()['traces']), 1)


if __name__ == '__main__':
    unittest.main()
