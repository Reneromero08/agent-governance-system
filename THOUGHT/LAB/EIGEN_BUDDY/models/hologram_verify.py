"""Holographic Phase Verification — operations encoded as phase signatures on S1."""
import torch, math

def verify():
    print("="*60)
    print("HOLOGRAM: OPERATION = PHASE SIGNATURE")
    print("="*60)
    
    # Operands
    for a,b in [(3,4),(7,2),(15,8),(25,17)]:
        # Encode: magnitude = value, phase = operation signature
        z_a_add = torch.polar(torch.tensor(a,dtype=torch.float32), torch.tensor(0.0))
        z_b_add = torch.polar(torch.tensor(b,dtype=torch.float32), torch.tensor(0.0))
        
        z_a_sub = torch.polar(torch.tensor(a,dtype=torch.float32), torch.tensor(0.0))
        z_b_sub = torch.polar(torch.tensor(b,dtype=torch.float32), torch.tensor(math.pi))
        
        z_a_mul = torch.polar(torch.tensor(a,dtype=torch.float32), torch.tensor(math.pi/2))
        z_b_mul = torch.polar(torch.tensor(b,dtype=torch.float32), torch.tensor(0.0))
        
        z_a_div = torch.polar(torch.tensor(a,dtype=torch.float32), torch.tensor(-math.pi/2))
        z_b_div = torch.polar(torch.tensor(b,dtype=torch.float32), torch.tensor(0.0))
        
        add = round((z_a_add + z_b_add).abs().item(), 2)
        sub = round((z_a_sub + z_b_sub).abs().item(), 2)
        mul = round((z_a_mul * z_b_mul.conj()).abs().item(), 2)
        div = round((z_a_div / z_b_div).abs().item(), 2)
        
        ok = (add==a+b and sub==a-b and mul==a*b)
        print(f"  a={a} b={b}: +={add} -={sub} *={mul} /= {div} {'OK' if ok else 'XX'}")
    
    # Test: does phase angle reveal the operation?
    print("\n--- Phase reads operation natively ---")
    for op_name,theta_op,theta_b in [("+",0.0,0.0),("-",0.0,math.pi),("*",math.pi/2,0.0),("/",-math.pi/2,0.0)]:
        za = torch.polar(torch.tensor(5.0), torch.tensor(theta_op))
        zb = torch.polar(torch.tensor(3.0), torch.tensor(theta_b))
        z_out = za + zb if op_name in "+-" else za * zb.conj()
        phase = torch.atan2(z_out.imag, z_out.real).item()
        print(f"  {op_name}: input phase=({theta_op:.2f},{theta_b:.2f}) output phase={phase:.3f} rad")

verify()
